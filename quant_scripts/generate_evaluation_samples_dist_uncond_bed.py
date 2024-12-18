"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
from PIL import Image
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'
import time
import logging

import numpy as np
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
import torch
from omegaconf import OmegaConf
from ldm_.util_ import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet, DDIMSampler_implicit_gaussian_quantCorrection_imagenet, DDIMSampler_integral_gaussian_quantCorrection_imagenet, DDIMSampler_gaussian_quantCorrection_imagenet, DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet, DDIMSampler_improved_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet
from quant_scripts.quant_dataset import DiffusionInputDataset, lsunInputDataset, get_calibration_set
from quant_scripts.brecq_uncond.brecq_quant_model_uncond import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer
import glob
import torch.nn as nn
# import pytorch_fid

n_bits_w = 4
n_bits_a = 8

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=25000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--correct_type', default='fp32')
    parser.add_argument('--local-rank', help="local device id on current node", type=int)
    parser.add_argument('--nproc_per_node', default=2, type=int)
    args = parser.parse_args()
    print(args)

    print('correct type: ', args.correct_type)


    ddim_steps = 200
    ddim_eta = 1.0
    data_type = 'bedroom'

    # init ddp
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.nproc_per_node, rank=local_rank)
    rank = torch.distributed.get_rank()
    
    # seed = int(time.time())
    seed = 100
    torch.manual_seed(seed + rank)
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    # Load model:
    base_configs = sorted(glob.glob('configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml'))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    gpu = True
    eval_mode = True

    print(config)
    ckpt='models/ldm/lsun_beds256/model.ckpt'
    model, global_step = load_model(config, ckpt=ckpt, gpu=gpu, eval_mode=eval_mode)
    print(f"global step: {global_step}")

    dmodel = model.model.diffusion_model
    dmodel.cuda(rank)
    model.cuda(rank)
    dmodel.eval()
    
    if args.correct_type != 'fp32':
        print('err!')
        wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
        aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
        qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
        qnn.cuda(rank)
        qnn.eval()

        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
        cali_images, cali_t = get_calibration_set('reproduce/lsun_{}_eta{}_step{}/data/image_input_{}.pth'.format(data_type, ddim_eta, ddim_steps, data_type), 'lsun')
        # cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
        print('cali_t: ', cali_t.shape)     
        device = next(qnn.parameters()).device
        # Initialize weight quantization parameters
        qnn.set_quant_state(True, True)

        print('First run to init model...')
        with torch.no_grad():
            _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device))
            
        # Start calibration
        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
                module.weight_quantizer.soft_targets = False
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        
        ckpt = torch.load('reproduce/lsun_{}_eta{}_step{}/w{}a{}/weights/quantw{}a{}_ldm_brecq_1000classes.pth'.format(data_type, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a), map_location='cpu')
        qnn.load_state_dict(ckpt)
        qnn.cuda(rank)
        qnn.eval()
        setattr(model.model, 'diffusion_model', qnn)

    model=nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])

    if args.correct_type == 'fp32':
        print('fp32!')
        sampler = DDIMSampler(model.module)
    elif args.correct_type == 'linear_correct':
        print('linear correct!')
        sampler = DDIMSampler_quantCorrection_imagenet(model.module, num_bit=4, correct=True)
    elif args.correct_type == 'linear_invariant_gaussian_correct':
        print('linear_invariant_gaussian_correct sampler!')
        sampler = DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True, scheme='linear_invariant')
    elif args.correct_type == 'linear_variable_gaussian_correctelse':
        print('linear_variable_gaussian_correctelse sampler!')
        sampler = DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True, scheme='linear_variable')
    elif args.correct_type == 'gaussian_correct':
        print('gaussian correct!')
        sampler = DDIMSampler_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True)
        # sampler = DDIMSampler_integral_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True)
    elif args.correct_type == 'implicit_gaussian_correct':
        sampler = DDIMSampler_implicit_gaussian_quantCorrection_imagenet(model.module, correct=True)
    elif args.correct_type == 'channel_wise_implicit_gaussian_correct':
        print('channel-wise implicit gaussian correct!')
        sampler = DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True)
    elif args.correct_type == 'channel_wise_explicit_gaussian_correct':
        print('channel-wise explicit gaussian correct!')
        sampler = DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet(model.module, num_bit=4, correct=True)

    save_dir = 'evaluate_data/lsun_{}_eta{}_step{}/'.format(data_type, ddim_eta, ddim_steps)
    if not os.path.exists(save_dir):
        print("make save dirs...")
        os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"{args.correct_type}_{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_type{data_type}_1.npz")
    print("out_path ", out_path)
    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        
        generated_num = torch.tensor(0, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)

    while generated_num.item() < args.num_samples:
        t0 = time.time()

        if args.correct_type == 'fp32':
            shape = [args.batch_size,
                 model.module.model.diffusion_model.in_channels,
                 model.module.model.diffusion_model.image_size,
                 model.module.model.diffusion_model.image_size]
        else:
            shape = [args.batch_size,
                 model.module.model.diffusion_model.model.in_channels,
                 model.module.model.diffusion_model.model.image_size,
                 model.module.model.diffusion_model.model.image_size]
        
        with model.module.ema_scope("Plotting"):
            t0 = time.time()
    
            bs = shape[0]
            shape = shape[1:]
            samples_ddim, _ = sampler.sample(ddim_steps, batch_size=bs, shape=shape, eta=ddim_eta, verbose=False,)
    

            t1 = time.time()

        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                    min=0.0, max=1.0)
        
        x_samples_ddim = (x_samples_ddim * 255.0).clamp(0, 255).to(torch.uint8)
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
        samples = x_samples_ddim.contiguous()

        t1 = time.time()
        print('throughput : {}'.format((t1 - t0) / x_samples_ddim.shape[0]))
        
        print('world_size: ', dist.get_world_size())
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  

        if rank == 0:
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            # print('dtype ', gathered_samples[0].cpu().numpy().dtype)
            logging.info(f"created {len(all_images)} samples")

            generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
            print('generated_num', generated_num.item())
                
        torch.distributed.barrier()
        dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()
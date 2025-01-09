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
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_implicit_gaussian_quantCorrection_imagenet, DDIMSampler_integral_gaussian_quantCorrection_imagenet, DDIMSampler_gaussian_quantCorrection_imagenet, DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet, DDIMSampler_improved_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet
from quant_scripts.dntc_quant_sample import dntc_sample
from quant_scripts.quant_dataset import DiffusionInputDataset, get_calibration_set
from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer
from quant_scripts.resample_calibration_data import resample_calibration

import torch.nn as nn
import pytorch_fid

n_bits_w = 4
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

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
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--correct_type', default='fp32')
    parser.add_argument('--local-rank', help="local device id on current node", type=int)
    parser.add_argument('--nproc_per_node', default=2, type=int)
    args = parser.parse_args()
    print(args)

    # out_dir = './evaluate_data'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # out_dir = None

    print('correct type: ', args.correct_type)

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0

    os.makedirs('evaluate_data/scale{}_eta{}_steps{}'.format(scale, ddim_eta, ddim_steps), exist_ok=True)

    # if args.correct_type == 'fp32':
    #     out_dir = 'evaluate_data/scale{}_eta{}_steps{}/fp32'.format(scale, ddim_eta, ddim_steps)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    # elif args.correct_type == 'gaussian_correct':
    #     out_dir = 'evaluate_data/scale{}_eta{}_steps{}/gaussian_correct'.format(scale, ddim_eta, ddim_steps)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    # elif args.correct_type == 'linear_invariant_gaussian_correct':
    #     out_dir = 'evaluate_data/scale{}_eta{}_steps{}/linear_invariant_gaussian_correct'.format(scale, ddim_eta, ddim_steps)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    # elif args.correct_type == 'linear_variable_gaussian_correct':
    #     out_dir = 'evaluate_data/scale{}_eta{}_steps{}/linear_variable_gaussian_correct'.format(scale, ddim_eta, ddim_steps)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    # elif args.correct_type == 'linear_correct':
    #     out_dir = 'evaluate_data/scale{}_eta{}_steps{}/linear_correct'.format(scale, ddim_eta, ddim_steps)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)

    # init ddp
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.nproc_per_node, rank=local_rank)
    rank = torch.distributed.get_rank()
    
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    # Load model:
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda(rank)
    model.cuda(rank)
    dmodel.eval()
    
    if args.correct_type != 'fp32':
        print('quantize model!')
        wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
        aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
        qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
        qnn.cuda(rank)
        qnn.eval()

        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
        
        dataset = DiffusionInputDataset('scale3.0_eta0.0_step20/imagenet/w4a8/imagenet_input_1000classes.pth')
        data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
        cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)

        # if resample calibration set
        # cali_images, cali_t, cali_y = resample_calibration('scale3.0_eta0.0_step20/imagenet/w4a8/imagenet_input_1000classes.pth')
    
        print('cali_t: ', cali_t.shape)     
        device = next(qnn.parameters()).device
        # Initialize weight quantization parameters
        qnn.set_quant_state(True, True)

        print('First run to init model...')
        with torch.no_grad():
            _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))
            
        # Start calibration
        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
                module.weight_quantizer.soft_targets = False
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        
        ckpt = torch.load('scale{}_eta{}_step{}/imagenet/w{}a{}/weights/quantw{}a{}_ldm_brecq_1000classes.pth'.format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a), map_location='cpu')
        print('load ckpt: scale{}_eta{}_step{}/imagenet/w{}a{}/weights/quantw{}a{}_ldm_brecq_1000classes.pth'.format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a))
        qnn.load_state_dict(ckpt)
        qnn.cuda(rank)
        qnn.eval()
        setattr(model.model, 'diffusion_model', qnn)

    model=nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])

    if args.correct_type == 'fp32':
        print(args.correct_type)
        sampler = DDIMSampler(model.module)
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

    out_path = os.path.join('evaluate_data/scale{}_eta{}_steps{}'.format(scale, ddim_eta, ddim_steps), f"{args.correct_type}_{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_scale{scale}_w{n_bits_w}a{n_bits_a}.npz")
    print("out_path ", out_path)
    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        all_labels = []
        
        generated_num = torch.tensor(0, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)
    n_samples_per_class = args.batch_size

    label_idx = 0
    while generated_num.item() < args.num_samples:

        if label_idx < 1000:
            class_labels = torch.tensor([min(label_idx+i, args.num_classes-1)for i in range(args.batch_size)], device=device)
            label_idx += args.batch_size
        else:
            class_labels = torch.randint(low=0,
                                        high=args.num_classes,
                                        size=(args.batch_size,),
                                        device=device)

        # class_labels = torch.randint(low=0,
        #                              high=args.num_classes,
        #                              size=(args.batch_size,),
        #                              device=device)

        print('class_labels ', class_labels)
        
        uc = model.module.get_learned_conditioning(
            {model.module.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.module.device)}
            )
        
        for class_label in class_labels:
            t0 = time.time()
            xc = torch.tensor(n_samples_per_class*[class_label]).to(model.module.device)
            c = model.module.get_learned_conditioning({model.module.cond_stage_key: xc.to(model.module.device)})
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples_per_class,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=ddim_eta)

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

            gathered_labels = [
                torch.zeros_like(xc) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, xc)

            if rank == 0:
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                print('dtype ', gathered_samples[0].cpu().numpy().dtype)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logging.info(f"created {len(all_images) * n_samples_per_class} samples")

                # save image
                # idx = 0
                # for i in range(len(gathered_samples)):
                #     bs = gathered_samples[i].shape[0]
                #     for j in range(bs):
                #         img_id = generated_num + idx
                #         img = gathered_samples[i][j].cpu().numpy()
                #         image_to_save = Image.fromarray(img.astype(np.uint8))
                #         image_to_save.save(out_dir + f"/img_{img_id :05d}.png")
                #         idx += 1

                generated_num = torch.tensor(len(all_images) * n_samples_per_class, device=device)
                
            torch.distributed.barrier()
            dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logging.info("sampling complete")
    print('label_idx: ', label_idx)


if __name__ == "__main__":
    main()

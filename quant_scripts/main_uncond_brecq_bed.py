import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
from quant_scripts.quant_dataset import DiffusionInputDataset, lsunInputDataset, get_calibration_set
from omegaconf import OmegaConf
from PIL import Image
sys.path.append('.')
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet
from ldm_.util_ import instantiate_from_config
from quant_scripts.brecq_uncond.brecq_quant_model_uncond import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer
import glob
from einops import rearrange
from torchvision.utils import make_grid

seed = 3343
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet(model, num_bit=4, correct=True)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.model.in_channels,
             model.model.diffusion_model.model.image_size,
             model.model.diffusion_model.model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
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


if __name__ == "__main__":
    n_bits_w = 4
    n_bits_a = 8
    data_type = 'bedroom'
    ddim_eta = 1.0
    ddim_steps = 200
    
    torch.cuda.set_device('cuda:0')
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
    dmodel.cuda()
    model.cuda()
    dmodel.eval()
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
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
    qnn.cuda()
    qnn.eval()
    setattr(model.model, 'diffusion_model', qnn)

    n_samples = 32

    batch_size = 4
    logdir = 'reproduce/lsun_church_eta0.0_step200/w4a8/generated_samples'

    print(f'Using DDIM sampling with {ddim_steps} sampling steps and eta={ddim_eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size, custom_steps=ddim_steps, eta=ddim_eta)
            x_samples_ddim = logs["sample"]
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 
                                            min=0.0, max=1.0)
            all_images.append(x_samples_ddim)
        # all_img = np.concatenate(all_images, axis=0)
        # all_img = all_img[:n_samples]
        # nppath = os.path.join(logdir, "samples.npz")
        # np.savez(nppath, all_img)
        
        grid = torch.stack(all_images, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=4)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        image_to_save = Image.fromarray(grid.astype(np.uint8))
        image_to_save.save("reproduce/lsun_church_eta0.0_step200/save_data/cwi_{}.png".format(seed))
        torch.save(all_images, "reproduce/lsun_church_eta0.0_step200/save_data/cwi_{}.pth".format(seed))

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    print("done.")
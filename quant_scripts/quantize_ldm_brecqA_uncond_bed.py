import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
# from taming.models import vqgan

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from ldm_.util_ import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm_.modules.diffusionmodules.openaimodel import ResBlock
from ldm_.modules.attention import SpatialTransformer, BasicTransformerBlock
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer
from quant_scripts.quant_dataset import get_calibration_set, lsunInputDataset, DiffusionInputDataset
from quant_scripts.brecq_uncond.brecq_quant_model_uncond import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_uncond.brecq_layer_recon_uncond import layer_reconstruction
from quant_scripts.brecq_uncond.brecq_block_recon_uncond import block_reconstruction_single_input, block_reconstruction_two_input
import glob
from tqdm import tqdm

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

global cnt 
cnt=0
def count_recon_times(model):
    global cnt
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                print('Ignore reconstruction of layer {}'.format(name))
                continue
            else:
                print('Reconstruction for layer {}'.format(name))
                cnt += 1
                # layer_reconstruction(qnn, module, **kwargs)


        elif isinstance(module, (ResBlock, BasicTransformerBlock)):
            print('Reconstruction for block {}'.format(name))
            cnt += 1
            # block_reconstruction(qnn, module, **kwargs)
        else:
            count_recon_times(module)

if __name__ == '__main__':
    torch.cuda.set_device('cuda:1')
    base_configs = sorted(glob.glob('configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml'))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    gpu = True
    eval_mode = True

    print(config)
    ckpt='models/LSUN_bedrooms/model.ckpt'
    model, global_step = load_model(config, ckpt=ckpt, gpu=gpu, eval_mode=eval_mode)
    print(f"global step: {global_step}")

    model = model.model.diffusion_model
    model.cuda()
    model.eval()

    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()
    cali_images, cali_t = get_calibration_set('reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom.pth', 'lsun')
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, iters=20000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', batch_size=32)
    
    save_dir = 'reproduce/lsun_bedroom_eta1.0_step200/w{}a{}/weights'.format(n_bits_w, n_bits_a)
    if not os.path.exists(save_dir):
        print("make save dirs...")
        os.makedirs(save_dir)
    
    pass_block = 0
    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        global pass_block
        for name, module in model.named_children():
            if isinstance(module, (QuantModule)):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)

            elif isinstance(module, ResBlock):
                pass_block -= 1
                if pass_block < 0 :
                    print('Reconstruction for ResBlock {}'.format(name))
                    block_reconstruction_two_input(qnn, module, **kwargs)
            elif isinstance(module, BasicTransformerBlock):
                pass_block -= 1
                if pass_block < 0 :
                    print('Reconstruction for BasicTransformerBlock {}'.format(name))
                    block_reconstruction_two_input(qnn, module, **kwargs)
            else:
                recon_model(module)
        
    # Start calibration
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    ckpt = torch.load(os.path.join(save_dir, 'quantw{}_ldm_brecq_1000classes.pth'.format(n_bits_w)), map_location='cpu') ## replace first step checkpoint here
    qnn.load_state_dict(ckpt, False)

    qnn.set_quant_state(True, True)
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device))
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()
    # Kwargs for activation rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, iters=5000, act_quant=True, opt_mode='mse', lr=4e-4, p=2.4, batch_size=16)
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    torch.save(qnn.state_dict(), os.path.join(save_dir, 'quantw{}a{}_ldm_brecq_1000classes.pth'.format(n_bits_w, n_bits_a)))
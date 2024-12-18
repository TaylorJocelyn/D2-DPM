import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import glob
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from quant_scripts.quant_dataset import DiffusionInputDataset, get_calibration_set
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler_collectQuantError
from tqdm import trange
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from quant_scripts.quant_dataset import lsunInputDataset
from quant_scripts.brecq_uncond.brecq_quant_model_uncond import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

n_bits_w = 4
n_bits_a = 8
# torch.cuda.manual_seed(100)

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

if __name__ == '__main__':
    torch.cuda.set_device('cuda:1')
    
    base_configs = sorted(glob.glob('configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml'))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    gpu = True
    eval_mode = True

    print(config)
    ckpt='models/LSUN_bedrooms/model.ckpt'
    fp_model, global_step = load_model(config, ckpt=ckpt, gpu=gpu, eval_mode=eval_mode)
    model, global_step = load_model(config, ckpt=ckpt, gpu=gpu, eval_mode=eval_mode)
    print(f"global step: {global_step}")

    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    
    data_type = 'bedroom'
    ddim_steps = 200
    ddim_eta = 1.0
    n_samples = 400
    batch_size = 16

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()
    cali_images, cali_t = get_calibration_set('reproduce/lsun_{}_eta{}_step{}/data/image_input_{}.pth'.format(data_type, ddim_eta, ddim_steps, data_type), 'lsun')
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

    ckpt = torch.load('reproduce/lsun_{}_eta{}_step{}/w{}a{}/weights/quantw{}a{}_ldm_brecq_1000classes.pth'.format(data_type, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a), map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()

    setattr(model.model, 'diffusion_model', qnn)

    sampler = DDIMSampler_collectQuantError(fp_model, model)

    print(f'Using DDIM sampling with {ddim_steps} sampling steps and eta={ddim_eta}')

    tstart = time.time()
    if model.cond_stage_model is None:
        # all_images = list()

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"): 
            shape = [batch_size,
            model.model.diffusion_model.model.in_channels,
            model.model.diffusion_model.model.image_size,
            model.model.diffusion_model.model.image_size]

            with model.ema_scope("Plotting"):
                t0 = time.time()

                bs = shape[0]
                shape = shape[1:]
                samples, intermediates = sampler.sample(ddim_steps, batch_size=batch_size, shape=shape, eta=ddim_eta, verbose=False,)
                t1 = time.time()

            # x_samples_ddim = model.decode_first_stage(samples)
            # x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
            #                                 min=0.0, max=1.0)
            # all_images.append(x_samples_ddim)

        import ldm.globalvar as globalvar
        data_error_t = globalvar.getList()
        torch.save(data_error_t, 'reproduce/lsun_{}_eta{}_step{}/w{}a{}/data_error_t_w{}a{}_eta{}_step{}_2.pth'.format(data_type, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a, ddim_eta, ddim_steps))
    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_samples} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    print("done.")



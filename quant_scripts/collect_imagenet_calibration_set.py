## To collect input data, remember to uncomment line 987-988 in ldm/models/diffusion/ddpm.py and comment them after finish collecting.
import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# from taming.models import vqgan

import random

import torch
from omegaconf import OmegaConf

from ldm_.util_ import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

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

if __name__ == '__main__':
    torch.cuda.set_device('cuda:1')
    model = get_model()
    sampler = DDIMSampler(model)


    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    classes = [i for i in range(1000)]

    # classes = random.sample(range(1000), 200)

    n_samples_per_class = 1

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0 

    all_samples = list()

    idx = 0 
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                print("-------- {} class ---------".format(idx))
                idx += 1
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    ## save diffusion input data
    import ldm_.globalvar as globalvar   
    input_list = globalvar.getInputList()

    save_dir = 'scale{}_eta{}_step{}/imagenet/w4a8/'.format(scale, ddim_eta, ddim_steps)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(input_list, save_dir + 'imagenet_input_{}classes.pth'.format(len(classes)))
    sys.exit(0)
 
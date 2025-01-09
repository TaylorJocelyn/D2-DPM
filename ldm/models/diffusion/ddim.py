"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import pickle

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2) # shape(12, 3, 64, 64)
            t_in = torch.cat([t] * 2) # shape(12)
            c_in = torch.cat([unconditional_conditioning, c]) # shape(12, 1, 512)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


class DDIMSampler_collectQuantError(object):
    def __init__(self, model, quant_model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.quant_model = quant_model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.hook_list = []

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # collect quantization error
        import ldm.globalvar as globalvar

        # if index==19:
        #     ## register hook for block corrector
        #     for name, module in self.quant_model.model.diffusion_model.named_modules():
        #         if isinstance(module, BlockOutputIdentity):
        #             self.hook_list.append(module.register_forward_hook(block_hook))

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            quant_e_t = self.quant_model.apply_model(x, t, c)

            globalvar.append((e_t, quant_e_t - e_t, t, quant_e_t))

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # globalvar.reset_cur_idx() ## this is for layerwise/blockwise error collection
            quant_e_t_uncond, quant_e_t = self.quant_model.apply_model(x_in, t_in, c_in).chunk(2)
            quant_e_t = quant_e_t_uncond + unconditional_guidance_scale * (quant_e_t - quant_e_t_uncond)
            globalvar.append((e_t, quant_e_t - e_t, t, quant_e_t))
            # globalvar.reset_cur_idx()
        
        # if index==0:
        #     for hook in self.hook_list:
        #         hook.remove()

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
# class DDIMSampler_gaussian_quantCorrection_imagenet(object):
#     def __init__(self, model, num_bit, correct=False, schedule="linear", **kwargs):
#         super().__init__()
#         self.model = model
#         self.correct  = correct
#         self.ddpm_num_timesteps = model.num_timesteps
#         self.schedule = schedule

#         self.mu_dict = np.load('correct_data/gaussians_250steps_w4a8/mu_dict.npy', allow_pickle=True).item()
#         self.cov_dict = np.load('correct_data/gaussians_250steps_w4a8/cov_dict.npy', allow_pickle=True).item()

#     def mu(self, t):
#         if self.correct:
#             int_t = t[0].item()
#             if int_t in self.mu_dict:
#                 return torch.tensor(self.mu_dict[int_t]).float().cuda()
#             else: 
#                 print('mu err!!!')
#                 return torch.zeros(2).float().cuda()
#         else:
#             print('mu err!!!')
#             return torch.zeros(2).float().cuda()

#     def cov(self, t):
#         if self.correct:
#             int_t = t[0].item()
#             if int_t in self.cov_dict:
#                 return torch.tensor(self.cov_dict[int_t]).float().cuda()
#             else:
#                 print('cov err!!!')
#                 return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
#         else:
#             print('cov err!!!')
#             return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
    
#     def register_buffer(self, name, attr):
#         if type(attr) == torch.Tensor:
#             if attr.device != torch.device("cuda"):
#                 attr = attr.to(torch.device("cuda"))
#         setattr(self, name, attr)

#     def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
#         self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
#                                                   num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
#         alphas_cumprod = self.model.alphas_cumprod
#         assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
#         to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

#         self.register_buffer('betas', to_torch(self.model.betas))
#         self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
#         self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
#         self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
#         self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
#         self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

#         # ddim sampling parameters
#         ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
#                                                                                    ddim_timesteps=self.ddim_timesteps,
#                                                                                    eta=ddim_eta,verbose=verbose)
#         self.register_buffer('ddim_sigmas', ddim_sigmas)
#         self.register_buffer('ddim_alphas', ddim_alphas)
#         self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
#         self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
#         sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
#             (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
#                         1 - self.alphas_cumprod / self.alphas_cumprod_prev))
#         self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

#     @torch.no_grad()
#     def sample(self,
#                S,
#                batch_size,
#                shape,
#                conditioning=None,
#                callback=None,
#                normals_sequence=None,
#                img_callback=None,
#                quantize_x0=False,
#                eta=0.,
#                mask=None,
#                x0=None,
#                temperature=1.,
#                noise_dropout=0.,
#                score_corrector=None,
#                corrector_kwargs=None,
#                verbose=True,
#                x_T=None,
#                log_every_t=100,
#                unconditional_guidance_scale=1.,
#                unconditional_conditioning=None,
#                # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
#                **kwargs
#                ):
#         if conditioning is not None:
#             if isinstance(conditioning, dict):
#                 cbs = conditioning[list(conditioning.keys())[0]].shape[0]
#                 if cbs != batch_size:
#                     print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
#             else:
#                 if conditioning.shape[0] != batch_size:
#                     print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

#         self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
#         # sampling
#         C, H, W = shape
#         size = (batch_size, C, H, W)
#         print(f'Data shape for DDIM sampling is {size}, eta {eta}')

#         samples, intermediates = self.ddim_sampling(conditioning, size,
#                                                     callback=callback,
#                                                     img_callback=img_callback,
#                                                     quantize_denoised=quantize_x0,
#                                                     mask=mask, x0=x0,
#                                                     ddim_use_original_steps=False,
#                                                     noise_dropout=noise_dropout,
#                                                     temperature=temperature,
#                                                     score_corrector=score_corrector,
#                                                     corrector_kwargs=corrector_kwargs,
#                                                     x_T=x_T,
#                                                     log_every_t=log_every_t,
#                                                     unconditional_guidance_scale=unconditional_guidance_scale,
#                                                     unconditional_conditioning=unconditional_conditioning,
#                                                     )
#         return samples, intermediates

#     @torch.no_grad()
#     def ddim_sampling(self, cond, shape,
#                       x_T=None, ddim_use_original_steps=False,
#                       callback=None, timesteps=None, quantize_denoised=False,
#                       mask=None, x0=None, img_callback=None, log_every_t=100,
#                       temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
#                       unconditional_guidance_scale=1., unconditional_conditioning=None,):
#         device = self.model.betas.device
#         b = shape[0]
#         if x_T is None:
#             img = torch.randn(shape, device=device)
#         else:
#             img = x_T

#         if timesteps is None:
#             timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
#         elif timesteps is not None and not ddim_use_original_steps:
#             subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
#             timesteps = self.ddim_timesteps[:subset_end]

#         intermediates = {'x_inter': [img], 'pred_x0': [img]}
#         time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
#         total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
#         print(f"Running DDIM Sampling with {total_steps} timesteps")

#         iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

#         for i, step in enumerate(iterator):
#             index = total_steps - i - 1
#             ts = torch.full((b,), step, device=device, dtype=torch.long)

#             if mask is not None:
#                 assert x0 is not None
#                 img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
#                 img = img_orig * mask + (1. - mask) * img
#             # print('--------------- t = ', step, ' ---------------') 
#             outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
#                                       quantize_denoised=quantize_denoised, temperature=temperature,
#                                       noise_dropout=noise_dropout, score_corrector=score_corrector,
#                                       corrector_kwargs=corrector_kwargs,
#                                       unconditional_guidance_scale=unconditional_guidance_scale,
#                                       unconditional_conditioning=unconditional_conditioning)
#             img, pred_x0 = outs
#             if callback: callback(i)
#             if img_callback: img_callback(pred_x0, i)

#             if index % log_every_t == 0 or index == total_steps - 1:
#                 intermediates['x_inter'].append(img)
#                 intermediates['pred_x0'].append(pred_x0)

#         return img, intermediates

#     @torch.no_grad()
#     def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
#                       temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
#                       unconditional_guidance_scale=1., unconditional_conditioning=None):
#         b, *_, device = *x.shape, x.device

#         if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
#             e_t = self.model.apply_model(x, t, c)
#             e_t = e_t

#         else: ## run here
#             x_in = torch.cat([x] * 2)
#             t_in = torch.cat([t] * 2)
#             c_in = torch.cat([unconditional_conditioning, c])
#             e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

#             e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
#             e_t = e_t

#         # e_t = e_t - self.saved_bias(index)
#         # e_t = e_t / (1+self.kt(t))
        
#         # sample error from conditional gaussian model
#         # normal_noise = np.random.standard_normal(e_t.shape)
#         normal_noise = torch.randn(e_t.shape).float().cuda()

#         # calculate conditional mean and std
#         covariance = self.cov(t)
#         mu = self.mu(t)

#         sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 1, 1, 1))
#         sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 1, 1, 1))
#         rho = torch.reshape(covariance[0, 1] / (sigma_X * sigma_Y), (1, 1, 1, 1))

#         rho_max = torch.tensor(0.9999).cuda()
#         rho_min = torch.tensor(-0.9999).cuda()

#         if covariance[0, 1] / rho_max >= sigma_X * sigma_Y:
#             rho = torch.reshape(rho_max, (1, 1, 1, 1))
#         if covariance[0, 1] / rho_min >= sigma_X * sigma_Y:
#             rho = torch.reshape(rho_min, (1, 1, 1, 1))

#         mu_y = torch.reshape(mu[1], (1, 1, 1, 1))
#         mu_x = torch.reshape(mu[0], (1, 1, 1, 1))

#         mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x)
#         # print('rho ', rho)
#         sigma_cond = sigma_Y * torch.sqrt(torch.ones(1,1,1,1).float().cuda() - rho**2)

#         quantized_error = mu_cond + normal_noise * sigma_cond

#         e_t = e_t - quantized_error

#         if score_corrector is not None: ## do not run
#             assert self.model.parameterization == "eps"
#             e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

#         alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
#         alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
#         sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
#         sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
#         # select parameters corresponding to the currently considered timestep
#         a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
#         a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
#         sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
#         sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

#         # current prediction for x_0
#         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
#         if quantize_denoised: ## do not run
#             pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
#         ## noise fusion
#         correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond
#         if 2 * torch.pow(correct_item[0],2) < torch.pow(sigma_t[0],2):
#             sigma_t = torch.sqrt(sigma_t**2 - 2 * torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond, 2))
#         else:
#             sigma_t = 0

#         # direction pointing to x_t
#         dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

#         noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
#         if noise_dropout > 0.:
#             noise = torch.nn.functional.dropout(noise, p=noise_dropout)
#         x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

#         # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
#         # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
#         # et_min, et_max = torch.min(e_t), torch.max(e_t)

#         # print("sigma_t ", sigma_t)
#         # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
#         # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
#         # print("e_t [ ", et_min, ' , ', et_max,' ]')

#         return x_prev, pred_x0

class DDIMSampler_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, num_bit, correct=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/w4a8/gaussian_correct/mu_dict.npy', allow_pickle=True).item()
        self.cov_dict = np.load('scale3.0_eta0.0_step20/w4a8/gaussian_correct/cov_dict.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else: 
                print('mu err!!!')
                return torch.zeros(2).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros(2).float().cuda()

    def cov(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t]).float().cuda()
            else:
                print('cov err!!!')
                return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
        else:
            print('cov err!!!')
            return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t

        # e_t = e_t - self.saved_bias(index)
        # e_t = e_t / (1+self.kt(t))
        
        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        covariance = self.cov(t)
        mu = self.mu(t)
   
        sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 1, 1, 1))
        sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 1, 1, 1))
        rho = torch.reshape(covariance[0, 1] / (sigma_X * sigma_Y), (1, 1, 1, 1))

        rho_max = torch.tensor(0.9999).cuda()
        rho_min = torch.tensor(-0.9999).cuda()

        if covariance[0, 1] / rho_max >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_max, (1, 1, 1, 1))
        if covariance[0, 1] / rho_min >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_min, (1, 1, 1, 1))

        mu_y = torch.reshape(mu[1], (1, 1, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 1, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x)
        # print('rho ', rho)
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1,1,1,1).float().cuda() - rho**2)

        quantized_error = mu_cond + normal_noise * sigma_cond

        e_t = e_t - quantized_error # shape(6, 3, 64, 64)

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
    # print("no std correct!")
        # noise fusion
        # correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond
        # if 2 * torch.pow(correct_item[0],2) < torch.pow(sigma_t[0],2):
            # sigma_t = torch.sqrt(sigma_t**2 - 2 * torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond, 2))
        # else:
            # sigma_t = 0

        # # noise fusion
        # coeff_2 = torch.pow(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())[0], 2) 
        # variance_overflow = 2 * coeff_2 * torch.pow(sigma_cond[0], 2) - 2 * coeff_2 * covariance[0][1]
        # if variance_overflow < torch.pow(sigma_t[0],2):
        #     sigma_t = torch.sqrt(sigma_t**2 - torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * (2 * torch.pow(sigma_cond, 2) - 2 * covariance[0][1]))
        # else:
        #     sigma_t = 0

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
        # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
        # et_min, et_max = torch.min(e_t), torch.max(e_t)

        # print("sigma_t ", sigma_t)
        # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
        # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
        # print("e_t [ ", et_min, ' , ', et_max,' ]')

        return x_prev, pred_x0

class DDIMSampler_integral_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, num_bit, correct=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/gaussian_correct/mu_dict_new.npy', allow_pickle=True).item()
        self.cov_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/gaussian_correct/cov_dict_new.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else: 
                print('mu err!!!')
                return torch.zeros(2).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros(2).float().cuda()

    def cov(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t][0]).float().cuda()
            else:
                print('cov err!!!')
                return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
        else:
            print('cov err!!!')
            return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
        
    def sigma_fpe(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t][1]).float().cuda()
            else:
                print('sigma_fpe err!!!')
                return torch.tensor(0).float().cuda()
        else:
            print('sigma_fpe err!!!')
            return torch.tensor(0).float().cuda()
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t

        # e_t = e_t - self.saved_bias(index)
        # e_t = e_t / (1+self.kt(t))
        
        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        covariance = self.cov(t)
        mu = self.mu(t)

        sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 1, 1, 1))
        sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 1, 1, 1))
        rho = torch.reshape(covariance[0, 1] / (sigma_X * sigma_Y), (1, 1, 1, 1))

        rho_max = torch.tensor(0.9999).cuda()
        rho_min = torch.tensor(-0.9999).cuda()

        if covariance[0, 1] / rho_max >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_max, (1, 1, 1, 1))
        if covariance[0, 1] / rho_min >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_min, (1, 1, 1, 1))

        mu_y = torch.reshape(mu[1], (1, 1, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 1, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x)
        # print('rho ', rho)
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1,1,1,1).float().cuda() - rho**2)

        quantized_error = mu_cond + normal_noise * sigma_cond

        e_t = e_t - quantized_error # shape(6, 3, 64, 64)

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        # ## noise fusion
        # correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond
        # if 2 * torch.pow(correct_item[0],2) < torch.pow(sigma_t[0],2):
        #     sigma_t = torch.sqrt(sigma_t**2 - 2 * torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond, 2))
        # else:
        #     sigma_t = 0

        # noise fusion
        cov_fpe = self.sigma_fpe(t)
        coeff_2 = torch.pow(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())[0], 2) 
        variance_overflow = 2 * coeff_2 * torch.pow(sigma_cond[0], 2) - 2 * coeff_2 * covariance[0][1] + 2 * coeff_2 * cov_fpe
        if variance_overflow < torch.pow(sigma_t[0],2):
            sigma_t = torch.sqrt(sigma_t**2 - torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * (2 * torch.pow(sigma_cond, 2) - 2 * covariance[0][1] + 2 * cov_fpe))
        else:
            sigma_t = 0

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
        # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
        # et_min, et_max = torch.min(e_t), torch.max(e_t)

        # print("sigma_t ", sigma_t)
        # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
        # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
        # print("e_t [ ", et_min, ' , ', et_max,' ]')

        return x_prev, pred_x0

    
class DDIMSampler_implicit_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, correct=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/w4a8/gaussian_correct/mu_dict.npy', allow_pickle=True).item()
        self.cov_dict = np.load('scale3.0_eta0.0_step20/w4a8/gaussian_correct/cov_dict.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else: 
                print('mu err!!!')
                return torch.zeros(2).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros(2).float().cuda()

    def cov(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t]).float().cuda()
            else:
                print('cov err!!!')
                return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
        else:
            print('cov err!!!')
            return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t

        # e_t = e_t - self.saved_bias(index)
        # e_t = e_t / (1+self.kt(t))
        
        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        # normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        covariance = self.cov(t)
        mu = self.mu(t)

        sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 1, 1, 1))
        sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 1, 1, 1))
        rho = torch.reshape(covariance[0, 1] / (sigma_X * sigma_Y), (1, 1, 1, 1))

        rho_max = torch.tensor(0.9999).cuda()
        rho_min = torch.tensor(-0.9999).cuda()

        if covariance[0, 1] / rho_max >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_max, (1, 1, 1, 1))
        if covariance[0, 1] / rho_min >= sigma_X * sigma_Y:
            rho = torch.reshape(rho_min, (1, 1, 1, 1))

        mu_y = torch.reshape(mu[1], (1, 1, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 1, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x)
        # print('rho ', rho)
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1,1,1,1).float().cuda() - rho**2)

        e_t = e_t - mu_cond
        # quantized_error = mu_cond + normal_noise * sigma_cond
        # e_t = e_t - quantized_error # shape(6, 3, 64, 64)

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        ## noise fusion
        correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond
        if torch.pow(correct_item[0],2) < torch.pow(sigma_t[0],2):
            sigma_t = torch.sqrt(sigma_t**2 - torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond, 2))
        else:
            sigma_t = 0

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
        # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
        # et_min, et_max = torch.min(e_t), torch.max(e_t)

        # print("sigma_t ", sigma_t)
        # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
        # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
        # print("e_t [ ", et_min, ' , ', et_max,' ]')

        return x_prev, pred_x0
    
class DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, num_bit, correct=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/channel_wise_gaussian_correct/mu_dict.npy', allow_pickle=True).item()
        self.cov_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/channel_wise_gaussian_correct/cov_dict.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else: 
                print('mu err!!!')
                return torch.zeros((2, 3)).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros((2, 3)).float().cuda()

    def cov(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t]).float().cuda()
            else:
                print('cov err!!!')
                return torch.tensor([
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]
                    ]
                ]).float().cuda()
        else:
            print('cov err!!!')
            return torch.tensor([
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]
                    ]
                ]).float().cuda()
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t

        # e_t = e_t - self.saved_bias(index)
        # e_t = e_t / (1+self.kt(t))
        
        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        covariance = self.cov(t) # sahpe(2, 2, 3)
        mu = self.mu(t) # shape(2, 3)

        sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 3, 1, 1))
        sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 3, 1, 1))
        rho = torch.reshape(covariance[0][1], (1, 3, 1, 1)) / (sigma_X * sigma_Y)

        rho_max = torch.tensor(0.9999).cuda()
        rho_min = torch.tensor(-0.9999).cuda()

        for ch in range(3):
            if (covariance[0][1][ch] / rho_max) >= (sigma_X[0][ch][0][0] * sigma_Y[0][ch][0][0]):
                rho[0][ch][0][0] = rho_max
            if (covariance[0][1][ch] / rho_min) >= (sigma_X[0][ch][0][0] * sigma_Y[0][ch][0][0]):
                rho[0][ch][0][0] = rho_min

        mu_y = torch.reshape(mu[1], (1, 3, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 3, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x) # shape(6, 3, 64, 64)
        # print('rho ', rho)
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1, 3, 1, 1).float().cuda() - rho**2) # shape(1, 3, 1, 1)

        quantized_error = mu_cond + normal_noise * sigma_cond # shape(6, 3, 64, 64)

        e_t = e_t - quantized_error # shape (6, 3, 64, 64) 

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 3, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        ## noise fusion
       #  correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond # shape(6, 3, 1, 1)
        # sigma_t = sigma_t.permute(1, 0, 2, 3)
        
        # for ch in range(3):
           # if 2 * torch.pow(correct_item[0][ch],2) < torch.pow(sigma_t[0][ch],2):
               # sigma_t[ch][:][:][:] = torch.reshape(torch.sqrt(sigma_t[0][ch]**2 - 2 * torch.pow((1. - a_prev - sigma_t[0][ch]**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond[0][ch], 2)), (b, 1, 1))
            # else:
                # sigma_t[ch][:][:][:] = torch.zeros(size=(b, 1, 1))

        # sigma_t = sigma_t.permute(1, 0, 2, 3)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
        # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
        # et_min, et_max = torch.min(e_t), torch.max(e_t)

        # print("sigma_t ", sigma_t)
        # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
        # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
        # print("e_t [ ", et_min, ' , ', et_max,' ]')
        return x_prev, pred_x0

class DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, num_bit, correct=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/channel_wise_gaussian_correct/mu_dict.npy', allow_pickle=True).item()
        
        self.cov_dict = np.load('scale3.0_eta0.0_step20/imagenet/w4a8/channel_wise_gaussian_correct/cov_dict.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else:
                print('mu err!!!')
                return torch.zeros((2, 3)).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros((2, 3)).float().cuda()

    def cov(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.cov_dict:
                return torch.tensor(self.cov_dict[int_t]).float().cuda()
            else:
                print('cov err!!!')
                return torch.tensor([
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]
                    ]
                ]).float().cuda()
        else:
            print('cov err!!!')
            return torch.tensor([
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]
                    ]
                ]).float().cuda()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t

        # e_t = e_t - self.saved_bias(index)
        # e_t = e_t / (1+self.kt(t))

        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        # normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        covariance = self.cov(t) # sahpe(2, 2, 3)
        mu = self.mu(t) # shape(2, 3)

        sigma_X = torch.reshape(torch.sqrt(covariance[0][0]), (1, 3, 1, 1))
        sigma_Y = torch.reshape(torch.sqrt(covariance[1][1]), (1, 3, 1, 1))
        rho = torch.reshape(covariance[0][1], (1, 3, 1, 1)) / (sigma_X * sigma_Y)

        rho_max = torch.tensor(0.9999).cuda()
        rho_min = torch.tensor(-0.9999).cuda()

        for ch in range(3):
            if (covariance[0][1][ch] / rho_max) >= (sigma_X[0][ch][0][0] * sigma_Y[0][ch][0][0]):
                rho[0][ch][0][0] = rho_max
            if (covariance[0][1][ch] / rho_min) >= (sigma_X[0][ch][0][0] * sigma_Y[0][ch][0][0]):
                rho[0][ch][0][0] = rho_min

        mu_y = torch.reshape(mu[1], (1, 3, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 3, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x) # shape(6, 3, 64, 64)
        # print('rho ', rho)
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1, 3, 1, 1).float().cuda() - rho**2) # shape(1, 3, 1, 1)

        # quantized_error = mu_cond + normal_noise * sigma_cond # shape(6, 3, 64, 64)

        e_t = e_t - mu_cond # shape (6, 3, 64, 64) 

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 3, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        ## noise fusion
        correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond # shape(6, 3, 1, 1)
        sigma_t = sigma_t.permute(1, 0, 2, 3)

        for ch in range(3):
            if torch.pow(correct_item[0][ch],2) < torch.pow(sigma_t[0][ch],2):
                sigma_t[ch][:][:][:] = torch.reshape(torch.sqrt(sigma_t[0][ch]**2 - torch.pow((1. - a_prev - sigma_t[0][ch]**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond[0][ch], 2)), (b, 1, 1))
            else:
                sigma_t[ch][:][:][:] = torch.zeros(size=(b, 1, 1))

        sigma_t = sigma_t.permute(1, 0, 2, 3)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

class DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet(object):
    def __init__(self, model, num_bit, correct=False, schedule="linear", scheme='linear_invariant', **kwargs):
        super().__init__()
        self.model = model
        self.correct  = correct
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.mu_dict = np.load('scale3.0_eta0.0_step20/imagenet/w8a8/gaussian_correct/mu_dict.npy', allow_pickle=True).item()
        self.cov_dict = np.load('scale3.0_eta0.0_step20/imagenet/w8a8/gaussian_correct/cov_dict.npy', allow_pickle=True).item()

        # self.quant_alpha = None
        # self.error_alpha = None

        self.scheme = scheme

        if self.scheme == 'linear_invariant':
            print('linear_invariant')
            with open('PTQD/reproduce/scale3.0_eta0.0_step20/imagenet/w8a8/variance_shrinkage/linear_invariant/quant_sigma_spline.pkl', 'rb') as f:
                self.quant_alpha = pickle.load(f)

            with open('PTQD/reproduce/scale3.0_eta0.0_step20/imagenet/w8a8/variance_shrinkage/linear_invariant/error_sigma_spline.pkl', 'rb') as f:
                self.error_alpha = pickle.load(f)

        elif self.scheme == 'linear_variable':
            print('linear_variable')
            with open('correct_data/variance_shrinkage/quant_sigma_spline_step250.pkl', 'rb') as f:
                self.quant_alpha = pickle.load(f)

            with open('correct_data/variance_shrinkage/error_sigma_spline_step250.pkl', 'rb') as f:
                self.error_alpha = pickle.load(f)

            self.rho_dict = np.load('observation_result/variance_shrinkage/gaussian2d_scheme1/rho_2.npy', allow_pickle=True).item()
        else:
            print('params_fully_variable')
            self.sigmax_dict = np.load('observation_result/variance_shrinkage/gaussian2d_scheme2/sigmax_2.npy', allow_pickle=True).item()
            self.sigmay_dict = np.load('observation_result/variance_shrinkage/gaussian2d_scheme2/sigmay_2.npy', allow_pickle=True).item()
            self.rho_dict = np.load('observation_result/variance_shrinkage/gaussian2d_scheme2/sigmax_2.npy', allow_pickle=True).item()

    def mu(self, t):
        if self.correct:
            int_t = t[0].item()
            if int_t in self.mu_dict:
                return torch.tensor(self.mu_dict[int_t]).float().cuda()
            else: 
                print('mu err!!!')
                return torch.zeros(2).float().cuda()
        else:
            print('mu err!!!')
            return torch.zeros(2).float().cuda()
        
    # def rho(self, t):
    #     if self.correct:
    #         int_t = t[0].item()
    #         if int_t in self.rho_dict:
    #             return torch.tensor(self.rho_dict[int_t]).float().cuda()
    #         else: 
    #             print('rho err!!!')
    #             return torch.ones(2).float().cuda()
    #     else:
    #         print('rho err!!!')
    #         return torch.ones(2).float().cuda()
        
    # def sigma_q(self, t):
    #     if self.correct:
    #         int_t = t[0].item()
    #         if int_t in self.cov_dict:
    #             return torch.tensor(self.quant_alpha(int_t) * np.sqrt(self.cov_dict[int_t][0][0])).float().cuda()
    #         else: 
    #             print('sigma_q err!!!')
    #             return torch.zeros(2).float().cuda()
    #     else:
    #         print('sigma_q err!!!')
    #         return torch.zeros(2).float().cuda()
        
    # def sigma_e(self, t):
    #     if self.correct:
    #         int_t = t[0].item()
    #         if int_t in self.cov_dict:
    #             return torch.tensor(self.quant_alpha(int_t) * np.sqrt(self.cov_dict[int_t][1][1])).float().cuda()
    #         else: 
    #             print('sigma_e err!!!')
    #             return torch.zeros(2).float().cuda()
    #     else:
    #         print('sigma_e err!!!')
    #         return torch.zeros(2).float().cuda()

    # def cov(self, t):
    #     if self.correct:
    #         int_t = t[0].item()
    #         if int_t in self.cov_dict:
    #                 covariance = self.cov_dict[int_t]
    #             if self.scheme == 'linear_invariant':
    #                 alpha_q = self.quant_alpha(int_t)
    #                 alpha_e = self.error_alpha(int_t)
    #                 covariance = np.multiply(covariance, np.array([[alpha_q**2, alpha_q * alpha_e], 
    #                                                                [alpha_q * alpha_e, alpha_e**2]]))
    #             elif self.scheme == 'linear_variable':

    #             else:
    #             # alpha_q = self.quant_alpha(int_t)
    #             # alpha_e = self.error_alpha(int_t)
    #             # sigma_x = np.sqrt(covariance[0][0])
    #             # sigma_y = np.sqrt(covariance[1][1])
    #             # rho = self.rho_dict[int_t]

    #             # vs_cov = np.array([[(alpha_q * sigma_x)**2, rho * sigma_x * sigma_y],
    #             #                    [rho * sigma_x * sigma_y, (alpha_e * sigma_y)**2]])

    #             # covariance = np.multiply(self.cov_dict[int_t], np.array([[alpha_q**2, alpha_q * alpha_e],
    #             #                                                          [alpha_q * alpha_e, alpha_e**2]]))
    #             return torch.tensor(covariance).float().cuda()
    #         else:
    #             print('cov err!!!')
    #             return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()
    #     else:
    #         print('cov err!!!')
    #         return torch.tensor([[1.0, 0.0],[0.0, 1.0]]).float().cuda()

    def get_cov_params(self, t):
        if self.correct:
            flag = False
            int_t = t[0].item()
            sigma_x, sigma_y, rho = None, None, None

            rho_max = 0.999999
            rho_min = -0.999999

            if self.scheme == 'linear_invariant':
                if int_t in self.cov_dict:
                    flag = True

                    covariance = self.cov_dict[int_t]
                    # alpha_q = min(1.0, self.quant_alpha(int_t))
                    # alpha_e = min(1.0, self.error_alpha(int_t))
                    alpha_q = self.quant_alpha(int_t)
                    alpha_e = self.error_alpha(int_t)
                    
                    sigma_x_unbiased = np.sqrt(covariance[0][0])
                    sigma_y_unbiased = np.sqrt(covariance[1][1])
                    sigma_x = sigma_x_unbiased * alpha_q
                    sigma_y = sigma_y_unbiased * alpha_e

                    if covariance[0][1] / rho_max >= sigma_x_unbiased * sigma_y_unbiased:
                        rho = rho_max
                    elif covariance[0][1] / rho_min >= sigma_x_unbiased * sigma_y_unbiased:
                        rho = rho_min
                    else:
                        rho = covariance[0][1] / (sigma_x_unbiased * sigma_y_unbiased)
            elif self.scheme == 'linear_variable':
                if int_t in self.cov_dict and int_t in self.rho_dict:
                    flag = True

                    covariance = self.cov_dict[int_t]
                    alpha_q = self.quant_alpha(int_t)
                    alpha_e = self.error_alpha(int_t)
                    
                    sigma_x = np.sqrt(covariance[0][0]) * alpha_q
                    sigma_y = np.sqrt(covariance[1][1]) * alpha_e
                    
                    rho = max(rho_min, min(self.rho_dict[int_t], rho_max))
            else:
                if int_t in self.sigmax_dict and int_t in self.sigmay_dict and int_t in self.rho_dict:
                    flag = True

                    sigma_x = self.sigmax_dict[int_t]
                    sigma_y = self.sigmay_dict[int_t]
                    rho = max(rho_min, min(self.rho_dict[int_t], rho_max))

            if not flag:
                print('flag False')
                return torch.ones(1).float().cuda(), torch.ones(1).float().cuda(), torch.zeros(1).float().cuda()
            else:
                sigma_x = torch.reshape(torch.tensor(sigma_x).float().cuda(), (1, 1, 1, 1))
                sigma_y = torch.reshape(torch.tensor(sigma_y).float().cuda(), (1, 1, 1, 1))
                rho = torch.reshape(torch.tensor(rho).float().cuda(), (1, 1, 1, 1))
                return sigma_x, sigma_y, rho
        else:
            print('get cov params err!!!')
            return torch.ones(1).float().cuda(), torch.ones(1).float().cuda(), torch.zeros(1).float().cuda()
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            # print('--------------- t = ', step, ' ---------------') 
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t

        else: ## run here
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t = e_t
        
        # sample error from conditional gaussian model
        # normal_noise = np.random.standard_normal(e_t.shape)
        normal_noise = torch.randn(e_t.shape).float().cuda()

        # calculate conditional mean and std
        mu = self.mu(t)
        sigma_X, sigma_Y, rho = self.get_cov_params(t)
        
        mu_y = torch.reshape(mu[1], (1, 1, 1, 1))
        mu_x = torch.reshape(mu[0], (1, 1, 1, 1))

        mu_cond = mu_y + rho * sigma_Y / sigma_X * (e_t - mu_x)
    
        sigma_cond = sigma_Y * torch.sqrt(torch.ones(1,1,1,1).float().cuda() - rho**2)

        quantized_error = mu_cond + normal_noise * sigma_cond

        e_t = e_t - quantized_error

        if score_corrector is not None: ## do not run
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        ## noise fusion
        correct_item = torch.abs(((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt())) * sigma_cond
        if 2 * torch.pow(correct_item[0],2) < torch.pow(sigma_t[0],2):
            sigma_t = torch.sqrt(sigma_t**2 - 2 * torch.pow((1. - a_prev - sigma_t**2).sqrt() - (a_prev*(1-a_t)/a_t).sqrt(), 2) * torch.pow(sigma_cond, 2))
        else:
            sigma_t = 0

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # x_prev_min, x_prev_max = torch.min(x_prev), torch.max(x_prev)
        # x0_pred_min, x0_pred_max = torch.min(pred_x0), torch.max(pred_x0)
        # et_min, et_max = torch.min(e_t), torch.max(e_t)

        # print("sigma_t ", sigma_t)
        # print("x_prev [ ", x_prev_min, ' , ', x_prev_max,' ]')
        # print("x0_pred [ ", x0_pred_min, ' , ', x0_pred_max,' ]')
        # print("e_t [ ", et_min, ' , ', et_max,' ]')

        return x_prev, pred_x0

<div align="center">
    <h1>D<sup>2</sup>-DPM: Dual Denoising for Quantized Diffusion Probabilistic Models</h1>
</div>

<div align="center">

  <a href="https://arxiv.org/abs/2501.08180"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2501.08180&color=B31B1B&logo=arxiv"></a>

</div>

# Overview
This is the official implementation of paper "D $^2$-DPM: Dual Denoising for Quantized Diffusion Probabilistic Models" [arXiv], which presents a dynamic quantization error correction strategy for diffusion models based on Gaussian modeling.

## Getting Started
### Requirements
Set up a virtual environment and install the required dependencies as specified in [LDM’s](https://github.com/CompVis/latent-diffusion) instructions.

### Sampling with the FP32 Model
<div style="padding-left: 0px;">
<pre>
python quant_scripts/sample_fp32_imagenet.py
</pre>
</div>


### Pre-trained models
Utilize the pre-trained models provided by LDM.

- For example, download the pre-trained LDM-4 model used in class-conditional generation experiments.
<div style="padding-left: 20px;">
<pre>
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt 
</pre>
</div>


### Pipeline
1. Collect an ImageNet calibration dataset with evenly spaced time interval sampling.
<div style="padding-left: 20px;">
<pre>
python quant_scripts/collect_imagenet_calibration_set.py
</pre>
</div>


2. Perform weight quantization using the BRECQ algorithm.
<div style="padding-left: 20px;">
<pre>
python quant_scripts/quantize_weight.py
</pre>
</div>


- *(Optional)* Optionally adjust the distribution of timesteps to generate calibration samples. 

  This requires:

  - Uncomment line 101 in quant_scripts/quantize_weight.py.
  - Comment out lines 96-98.

3. Perform activation quantization using the BRECQ algorithm.
<div style="padding-left: 20px;">
<pre>
python quant_scripts/quantize_weight_activation.py
</pre>
</div>

- *(Optional)* Optionally adjust the distribution of timesteps to generate calibration samples, ensuring consistency with the selection in **2.**

  This requires:

  - Uncomment line 101 in quant_scripts/quantize_weight.py.
  - Comment out lines 95-98.

4. Collect statistics for timestep-aware Gaussian modeling in preparation for the next step.
<div style="padding-left: 20px;">
<pre>
python quant_scripts/collect_timestep_aware_statistics.py
</pre>
</div>

5. Perform timestep-aware Gaussian modeling, optionally choosing between tensor-wise or channel-wise modeling.

<div style="padding-left: 20px;">
<pre>
python quant_scripts/gaussian_modeling.py
</pre>
</div>

6. Sample using the quantized model and corrected sampler, selecting from the following corrected samplers based on the provided descriptions:
 - **DDIMSampler_gaussian_quantCorrection_imagenet:**
   
    *Utilizes tensor-wise modeling and applies the S-D<sup>2</sup> denoising strategy.*

 - **DDIMSampler_implicit_gaussian_quantCorrection_imagenet:**
   
    *Utilizes tensor-wise modeling and applies the D-D<sup>2</sup> denoising strategy.*

 - **DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet:**
   
    *Utilizes channel-wise modeling and applies the S-D<sup>2</sup> denoising strategy.*

 - **DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet:**
   
    *Utilizes channel-wise modeling and applies the D-D<sup>2</sup> denoising strategy.*

<div style="padding-left: 20px;">
<pre>
python quant_scripts/sample_quantized_ldm_imagenet.py
</pre>
</div>

### Evaluation

1. Generate at least 50,000 samples. You may also choose whether to perform calibration dataset resampling.

<div style="padding-left: 20px;">
<pre>
python quant_scripts/generate_evaluation_samples_dist.py
</pre>
</div>

2. Evaluate using OpenAI’s evaluator and the open-source reference batch available at [guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations).

<div style="padding-left: 20px;">
<pre>
python evaluator.py --ref_batch VIRTUAL_imagenet256_labeled.npz --sample_batch your_synthesized_samples.npz
</pre>
</div>

## Acknowledgement

This repository is built upon [LDM](https://github.com/CompVis/latent-diffusion) and [PTQD](https://github.com/ziplab/PTQD). We thank the authors for their open-sourced code.

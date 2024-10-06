import contextlib
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from transformers.utils import is_peft_available

from trl.core import randn_tensor
from trl.models.sd_utils import convert_state_dict_to_diffusers
from trl.models.modeling_sd_base import DefaultDDPOStableDiffusionPipeline, DDPOPipelineOutput, DDPOSchedulerOutput, scheduler_step

if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict


def pipeline_step_with_grad(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    gradient_checkpoint: bool = True,
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function to get RGB image with gradients attached to the model weights.

    Args:
        prompt (`str` or `List[str]`, *optional*, defaults to `None`):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.
        height (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to `7.5`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        truncated_backprop (`bool`, *optional*, defaults to True):
            Truncated Backpropation to fixed timesteps, helps prevent collapse during diffusion reward training as shown in AlignProp (https://huggingface.co/papers/2310.03739).
        truncated_backprop_rand (`bool`, *optional*, defaults to True):
            Truncated Randomized Backpropation randomizes truncation to different diffusion timesteps, this helps prevent collapse during diffusion reward training as shown in AlignProp (https://huggingface.co/papers/2310.03739).
            Enabling truncated_backprop_rand allows adapting earlier timesteps in diffusion while not resulting in a collapse.
        gradient_checkpoint (`bool`, *optional*, defaults to True):
            Adds gradient checkpointing to Unet forward pass. Reduces GPU memory consumption while slightly increasing the training time.
        truncated_backprop_timestep (`int`, *optional*, defaults to 49):
            Absolute timestep to which the gradients are being backpropagated. Higher number reduces the memory usage and reduces the chances of collapse.
            While a lower value, allows more semantic changes in the diffusion generations, as the earlier diffusion timesteps are getting updated.
            However it also increases the chances of collapse.
        truncated_rand_backprop_minmax (`Tuple`, *optional*, defaults to (0,50)):
            Range for randomized backprop. Here the value at 0 index indicates the earlier diffusion timestep to update (closer to noise), while the value
            at index 1 indicates the later diffusion timestep to update.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `pipeline.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        `DDPOPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities
    """

    backprop_timestep = -1
    
    while backprop_timestep >= num_inference_steps or backprop_timestep < 1:    
        if backprop_strategy == 'gaussian':
            backprop_timestep = int(torch.distributions.Normal(backprop_kwargs['mean'], backprop_kwargs['std']).sample().item())
        elif backprop_strategy == 'uniform':
            backprop_timestep = int(torch.randint(backprop_kwargs['min'], backprop_kwargs['max'], (1,)).item())
        elif backprop_strategy == 'fixed':
            backprop_timestep = int(backprop_kwargs['value'])
    
    if height is None and width is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if gradient_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    pipeline.unet,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    use_reentrant=False,
                )[0]
            else:
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            
            if i < backprop_timestep:
                noise_pred = noise_pred.detach()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents, eta)
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = pipeline.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    return DDPOPipelineOutput(image, all_latents, all_log_probs)


class DiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, pretrained_model_name: str, pretrained_model_revision: str = "main", use_lora: bool = True):
        super().__init__(pretrained_model_name,pretrained_model_revision=pretrained_model_revision,use_lora=use_lora)

    def rgb_with_grad(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step_with_grad(self.sd_pipeline, *args, **kwargs)    
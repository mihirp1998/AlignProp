import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from aesthetic_scorer import AestheticScorerDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/align_prop.py", "Training configuration.")
from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)


def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    checkpoint_path = "/home/mprabhud/.cache/hpsv2/HPS_v2_compressed.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
    

def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn



def evaluate(latent,train_neg_prompt_embeds,prompts, pipeline, accelerator, inference_dtype, config, loss_fn):
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)       
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
    

    all_rgbs_t = []
    for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), total=len(pipeline.scheduler.timesteps)):
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = pipeline.unet(latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = pipeline.unet(latent, t, prompt_embeds).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
    if "hps" in config.reward_fn:
        loss, rewards = loss_fn(ims, prompts)
    else:    
        _, rewards = loss_fn(ims)
    return ims, rewards

    
    

def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    
    if accelerator.is_main_process:
        wandb_args = {}
        if config.debug:
            wandb_args = {'mode':"disabled"}        
        accelerator.init_trackers(
            project_name="align-prop", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )
        import wandb
        accelerator.project_configuration.project_dir = os.path.join(config.logdir, wandb.run.name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    
    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)


    # disable safety checker
    pipeline.safety_checker = None    
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.    
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16    

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    pipeline.unet.to(accelerator.device, dtype=inference_dtype)    
    # Set correct lora layers
    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    pipeline.unet.set_attn_processor(lora_attn_procs)

    # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
    # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
    # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipeline.unet(*args, **kwargs)

    unet = _Wrapper(pipeline.unet.attn_processors)        

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        output_splits = output_dir.split("/")
        output_splits[1] = wandb.run.name
        output_dir = "/".join(output_splits)
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.soup_inference:
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            if config.resume_from_2 != "same":
                tmp_unet_2 = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
                )
                tmp_unet_2.load_attn_procs(config.resume_from_2)
                
                attn_state_dict_2 = AttnProcsLayers(tmp_unet_2.attn_processors).state_dict()
                
            attn_state_dict = AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            if config.resume_from_2 == "same":
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1
            else:
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1 + attn_state_dict_2[attn_state_key]*(1.0 - config.mixing_coef_1)
            
            models[0].load_state_dict(attn_state_dict)
                    
            del tmp_unet                
            
            if config.resume_from_2 != "same":
                del tmp_unet_2
                
        elif isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)    

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW


    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_fn = getattr(prompts_file, config.prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext
    
    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)
    
    if config.reward_fn=='hps':
        loss_fn = hps_loss_fn(inference_dtype, accelerator.device)
    elif config.reward_fn=='aesthetic': # easthetic
        loss_fn = aesthetic_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
    else:
        raise NotImplementedError

    keep_input = True
    timesteps = pipeline.scheduler.timesteps
    
    eval_prompts, eval_prompt_metadata = zip(
        *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
    )    

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0 
       
    global_step = 0

    if config.only_eval:
        #################### EVALUATION ONLY ####################                
        import wandb
        all_eval_images = []
        all_eval_rewards = []
        if config.same_evaluation:
            generator = torch.cuda.manual_seed(config.seed)
            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
        else:
            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)        
        with torch.no_grad():
            for index in range(config.max_vis_images):
                ims, rewards = evaluate(latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],train_neg_prompt_embeds, eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], pipeline, accelerator, inference_dtype,config, loss_fn)
                all_eval_images.append(ims)
                all_eval_rewards.append(rewards)
        eval_rewards = torch.cat(all_eval_rewards)
        eval_reward_mean = eval_rewards.mean()
        print("Evaluation results", eval_reward_mean)
        eval_images = torch.cat(all_eval_images)
        eval_image_vis = []
        if accelerator.is_main_process:
            import wandb
            if config.run_name != "":
                name_val = config.run_name
            else:
                name_val = wandb.run.name            
            log_dir = f"logs/{name_val}/eval_vis"
            os.makedirs(log_dir, exist_ok=True)
            for i, eval_image in enumerate(eval_images):
                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                prompt = eval_prompts[i]
                pil.save(f"{log_dir}/{i:03d}_{prompt}.png")
                pil = pil.resize((256, 256))
                reward = eval_rewards[i]
                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
            accelerator.log({"eval_images": eval_image_vis},step=global_step)        
    else:
        #################### TRAINING ####################        
        for epoch in list(range(first_epoch, config.num_epochs)):
            unet.train()
            info = defaultdict(list)
            info_vis = defaultdict(list)
            image_vis_list = []
            
            for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_local_main_process):
                latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)    

                if accelerator.is_main_process:
                    import wandb
                    logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")

                
                prompts, prompt_metadata = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )

                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)   

                pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
                
            
                with accelerator.accumulate(unet):
                    with autocast():
                        with torch.enable_grad(): # important b/c don't have on by default in module                        
                            keep_input = True
                            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                                t = torch.tensor([t],
                                                    dtype=inference_dtype,
                                                    device=latent.device)
                                t = t.repeat(config.train.batch_size_per_gpu_available)
                                
                                if config.grad_checkpoint:
                                    noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                    noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                                else:
                                    noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                                    noise_pred_cond = unet(latent, t, prompt_embeds).sample
                                                                
                                if config.truncated_backprop:
                                    if config.truncated_backprop_rand:
                                        timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                        if i < timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                    else:
                                        if i < config.trunc_backprop_timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()

                                grad = (noise_pred_cond - noise_pred_uncond)
                                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad                
                                latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                                    
                            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                            
                            if "hps" in config.reward_fn:
                                loss, rewards = loss_fn(ims, prompts)
                            else:
                                loss, rewards = loss_fn(ims)
                            
                            loss =  loss.sum()
                            loss = loss/config.train.batch_size_per_gpu_available
                            loss = loss * config.train.loss_coeff
                            

                            rewards_mean = rewards.mean()
                            rewards_std = rewards.std()
                            
                            if len(info_vis["image"]) < config.max_vis_images:
                                info_vis["image"].append(ims.clone().detach())
                                info_vis["rewards_img"].append(rewards.clone().detach())
                                info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                            
                            info["loss"].append(loss)
                            info["rewards"].append(rewards_mean)
                            info["rewards_std"].append(rewards_std)
                            
                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()                        

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0
                    # log training and evaluation 
                    if config.visualize_eval and (global_step % config.vis_freq ==0):
                        import wandb
                        all_eval_images = []
                        all_eval_rewards = []
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                        else:
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)                                
                        with torch.no_grad():
                            for index in range(config.max_vis_images):
                                ims, rewards = evaluate(latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],train_neg_prompt_embeds, eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], pipeline, accelerator, inference_dtype,config, loss_fn)
                                all_eval_images.append(ims)
                                all_eval_rewards.append(rewards)
                        eval_rewards = torch.cat(all_eval_rewards)
                        eval_reward_mean = eval_rewards.mean()
                        eval_reward_std = eval_rewards.std()
                        eval_images = torch.cat(all_eval_images)
                        eval_image_vis = []
                        if accelerator.is_main_process:
                            import wandb
                            name_val = wandb.run.name
                            log_dir = f"logs/{name_val}/eval_vis"
                            os.makedirs(log_dir, exist_ok=True)
                            for i, eval_image in enumerate(eval_images):
                                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                                pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                                prompt = eval_prompts[i]
                                pil.save(f"{log_dir}/{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                                pil = pil.resize((256, 256))
                                reward = eval_rewards[i]
                                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
                            accelerator.log({"eval_images": eval_image_vis},step=global_step)
                    
                    logger.info("Logging")
                    
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                    info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
                    accelerator.log(info, step=global_step)

                    if config.visualize_train:
                        ims = torch.cat(info_vis["image"])
                        rewards = torch.cat(info_vis["rewards_img"])
                        prompts = info_vis["prompts"]
                        images  = []
                        for i, image in enumerate(ims):
                            image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            pil = pil.resize((256, 256))
                            prompt = prompts[i]
                            reward = rewards[i]
                            images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                        
                        accelerator.log(
                            {"images": images},
                            step=global_step,
                        )

                    global_step += 1
                    info = defaultdict(list)


            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            
            if epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()


                        
                        
                        


if __name__ == "__main__":
    app.run(main)
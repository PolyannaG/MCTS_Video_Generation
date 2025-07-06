# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        use_exact_token_counting=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            use_exact_token_counting (`bool`, *optional*, defaults to False):
                Whether to use exact token counting with the tokenizer instead of approximation.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.prev_context_prompts = [
            "CONTEXT (previous chunk): ",
            "IMPORTANT: The above context is only for continuity information. DO NOT include elements from it directly.",
            "GENERATE THIS: "
        ]
        # self.prev_context_prompts = [
        #     "The previously generated video used the prompt: ",
        #     "The video we are generating now is a continuation of the previous video.",
        #     "The prompt for the current video is: "
        # ]
        
        # Whether to use exact token counting with tokenizer instead of approximation
        self.use_exact_token_counting = use_exact_token_counting

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def generate_multi(self,
                 input_prompt,
                 imgs,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from multiple input images and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            imgs (list[PIL.Image.Image]):
                List of input image frames. Each should be a PIL Image.
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # If only one image is provided, use the original generate method
        if len(imgs) == 1:
            return self.generate(
                input_prompt=input_prompt,
                img=imgs[0],
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
            
        # Convert PIL images to tensors and normalize
        img_tensors = [TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device) for img in imgs]
        
        # Get dimensions from the last frame (which will be used for latent space preservation)
        last_frame = img_tensors[-1]
        
        F = frame_num
        h, w = last_frame.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # Create mask - ONLY mark the first position for preservation
        # This ensures only the last frame's latent is preserved in the first position
        msk = torch.zeros(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, 0] = 1  # Only preserve the first frame position
        
        # Format mask to match model expectations (chunk into groups of 4 frames)
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Text encoding
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Process all conditioning frames with CLIP for feature guidance
        self.clip.model.to(self.device)
        
        # Extract CLIP features from all frames and combine them
        clip_features = []
        for img_tensor in img_tensors:
            # Extract features for each frame
            feature = self.clip.visual([img_tensor[:, None, :, :]])
            clip_features.append(feature)
        
        # Combine CLIP features (average them)
        combined_clip_features = torch.stack(clip_features).mean(dim=0)
        
        if offload_model:
            self.clip.model.cpu()

        # Resize all conditioning frames to target size
        resized_tensors = []
        for img_tensor in img_tensors:
            resized = torch.nn.functional.interpolate(
                img_tensor[None].cpu(), size=(h, w), mode='bicubic'
            ).transpose(0, 1).to(self.device)
            resized_tensors.append(resized)
        
        # Create input tensor with the last frame in the first position
        # for temporal continuity and latent space preservation
        input_tensor = torch.zeros(3, F, h, w, device=self.device)
        
        # Place ONLY the last frame in the first position
        input_tensor[:, 0:1, :, :] = resized_tensors[-1]
        
        # VAE encode the input tensor
        y = self.vae.encode([input_tensor])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            # Use the combined CLIP features from all frames for conditioning
            arg_c = {
                'context': [context[0]],
                'clip_fea': combined_clip_features,  # Combined features from all frames
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': combined_clip_features,  # Combined features from all frames
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def generate_with_context(self,
                 input_prompt,
                 context_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt, using context prompts
        from previous chunks for better continuity without including their content directly.

        Args:
            input_prompt (`str`):
                Text prompt for content generation (current chunk).
            context_prompt (`str` or `list`):
                Context prompts from previous chunks (for continuity only). Can be a single string or list of strings.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W)
        """
        # If no context prompt is provided, fallback to the regular generate method
        if context_prompt is None or (isinstance(context_prompt, str) and context_prompt == "") or (isinstance(context_prompt, list) and len(context_prompt) == 0):
            return self.generate(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
        
        # Convert single context prompt to list if necessary
        context_prompts = context_prompt if isinstance(context_prompt, list) else [context_prompt]
        
        # Reverse the list to prioritize recent prompts
        context_prompts = context_prompts[::-1]
        
        if self.use_exact_token_counting:
            # Use exact token counting with tokenizer
            # Create a complete prompt with all formatting to check total available tokens
            all_format_elements = " ".join([
                self.prev_context_prompts[0],
                self.prev_context_prompts[1],
                self.prev_context_prompts[2]
            ])
            
            # Get token count for the current prompt and formatting
            tokenizer = self.text_encoder.tokenizer
            format_token_count = len(tokenizer(all_format_elements)[0])
            current_prompt_token_count = len(tokenizer(input_prompt)[0])
            
            # Reserve space for current prompt and formatting (add small buffer for safety)
            buffer_tokens = 20
            available_tokens = self.config.text_len - current_prompt_token_count - format_token_count - buffer_tokens
            
            # Select as many prompts as possible starting from most recent
            selected_prompts = []
            accumulated_tokens = 0
            
            for prompt in context_prompts:
                # Get exact token count for this prompt
                prompt_token_count = len(tokenizer(prompt)[0])
                
                # Check if adding this prompt would exceed the limit
                if accumulated_tokens + prompt_token_count > available_tokens:
                    break
                    
                selected_prompts.append(prompt)
                accumulated_tokens += prompt_token_count
        else:
            # Use approximate token counting (original method)
            estimated_tokens = {
                'current_prompt': len(input_prompt.split()) * 1.5,  # Rough estimate: 1.5 tokens per word
                'formatting': len(self.prev_context_prompts[0].split() + 
                                self.prev_context_prompts[1].split() + 
                                self.prev_context_prompts[2].split()) * 1.5,
                'buffer': 50,  # Buffer for tokenization differences and special tokens
            }
            
            # Reserve space for current prompt and formatting
            available_tokens = self.config.text_len - estimated_tokens['current_prompt'] - estimated_tokens['formatting'] - estimated_tokens['buffer']
            
            # Select as many prompts as possible starting from the most recent
            selected_prompts = []
            accumulated_tokens = 0
            
            for prompt in context_prompts:
                # Estimate token count for this prompt
                prompt_tokens = len(prompt.split()) * 1.5
                
                # Check if adding this prompt would exceed the limit
                if accumulated_tokens + prompt_tokens > available_tokens:
                    break
                    
                selected_prompts.append(prompt)
                accumulated_tokens += prompt_tokens
        
        # Reverse back to chronological order
        selected_prompts = selected_prompts[::-1]
        
        # If no context prompts can be included after token limiting, fall back to the normal generate method
        if len(selected_prompts) == 0:
            return self.generate(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
        
        # Concatenate all selected context prompts with separators
        combined_context = " ".join(selected_prompts)
            
        # Create the formatted prompt with the selected context
        formatted_prompt = f"{self.prev_context_prompts[0]} {combined_context} {self.prev_context_prompts[1]} {input_prompt} {self.prev_context_prompts[2]}"
            
        # Now use the formatted prompt with the standard generate method
        return self.generate(
            input_prompt=formatted_prompt,
            img=img,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model
        )

    def generate_multi_with_context(self,
                 input_prompt,
                 context_prompt,
                 imgs,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from multiple input images and text prompt,
        using context prompts from previous chunks for better continuity.

        Args:
            input_prompt (`str`):
                Text prompt for content generation (current chunk).
            context_prompt (`str` or `list`):
                Context prompts from previous chunks (for continuity only). Can be a single string or list of strings.
            imgs (list[PIL.Image.Image]):
                List of input image frames. Each should be a PIL Image.
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W)
        """
        # If only one image is provided, use the standard context method
        if len(imgs) == 1:
            return self.generate_with_context(
                input_prompt=input_prompt,
                context_prompt=context_prompt,
                img=imgs[0],
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
            
        # If no context prompt is provided, use the regular multi-frame method
        if context_prompt is None or (isinstance(context_prompt, str) and context_prompt == "") or (isinstance(context_prompt, list) and len(context_prompt) == 0):
            return self.generate_multi(
                input_prompt=input_prompt,
                imgs=imgs,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
        
        # Convert single context prompt to list if necessary
        context_prompts = context_prompt if isinstance(context_prompt, list) else [context_prompt]
        
        # Reverse the list to prioritize recent prompts
        context_prompts = context_prompts[::-1]
        
        if self.use_exact_token_counting:
            # Use exact token counting with tokenizer
            # Create a complete prompt with all formatting to check total available tokens
            all_format_elements = " ".join([
                self.prev_context_prompts[0],
                self.prev_context_prompts[1],
                self.prev_context_prompts[2]
            ])
            
            # Get token count for the current prompt and formatting
            tokenizer = self.text_encoder.tokenizer
            format_token_count = len(tokenizer(all_format_elements)[0])
            current_prompt_token_count = len(tokenizer(input_prompt)[0])
            
            # Reserve space for current prompt and formatting (add small buffer for safety)
            buffer_tokens = 20
            available_tokens = self.config.text_len - current_prompt_token_count - format_token_count - buffer_tokens
            
            # Select as many prompts as possible starting from most recent
            selected_prompts = []
            accumulated_tokens = 0
            
            for prompt in context_prompts:
                # Get exact token count for this prompt
                prompt_token_count = len(tokenizer(prompt)[0])
                
                # Check if adding this prompt would exceed the limit
                if accumulated_tokens + prompt_token_count > available_tokens:
                    break
                    
                selected_prompts.append(prompt)
                accumulated_tokens += prompt_token_count
        else:
            # Use approximate token counting (original method)
            estimated_tokens = {
                'current_prompt': len(input_prompt.split()) * 1.5,  # Rough estimate: 1.5 tokens per word
                'formatting': len(self.prev_context_prompts[0].split() + 
                                self.prev_context_prompts[1].split() + 
                                self.prev_context_prompts[2].split()) * 1.5,
                'buffer': 50,  # Buffer for tokenization differences and special tokens
            }
            
            # Reserve space for current prompt and formatting
            available_tokens = self.config.text_len - estimated_tokens['current_prompt'] - estimated_tokens['formatting'] - estimated_tokens['buffer']
            
            # Select as many prompts as possible starting from the most recent
            selected_prompts = []
            accumulated_tokens = 0
            
            for prompt in context_prompts:
                # Estimate token count for this prompt
                prompt_tokens = len(prompt.split()) * 1.5
                
                # Check if adding this prompt would exceed the limit
                if accumulated_tokens + prompt_tokens > available_tokens:
                    break
                    
                selected_prompts.append(prompt)
                accumulated_tokens += prompt_tokens
        
        # Reverse back to chronological order
        selected_prompts = selected_prompts[::-1]
        
        # If no context prompts can be included after token limiting, fall back to the normal generate method
        if len(selected_prompts) == 0:
            return self.generate_multi(
                input_prompt=input_prompt,
                imgs=imgs,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
        
        # Concatenate all selected context prompts with separators
        combined_context = " ".join(selected_prompts)
            
        # Create the formatted prompt with the selected context
        formatted_prompt = f"{self.prev_context_prompts[0]} {combined_context} {self.prev_context_prompts[1]} {input_prompt} {self.prev_context_prompts[2]}"
            
        # Use the formatted prompt with the multi-frame generation method
        return self.generate_multi(
            input_prompt=formatted_prompt,
            imgs=imgs,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model
        )

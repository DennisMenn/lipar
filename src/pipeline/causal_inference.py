from typing import List, Optional, Tuple
import logging
import torch

from src.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from src.utils.demo_utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
)

from src.utils.rlt_prune import rlt_prune


logger = logging.getLogger("self_forcing.pipeline.causal_inference")


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparameters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_cache(self, batch_size, dtype, device, kv_cache_size=None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if kv_cache_size is None:
            if self.local_attn_size != -1:
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                kv_cache_size = 32760

        can_reuse_cache = (
            self.kv_cache1 is not None
            and len(self.kv_cache1) == self.num_transformer_blocks
            and self.kv_cache1[0]["k"].shape[0] == batch_size
            and self.kv_cache1[0]["k"].shape[1] == kv_cache_size
            and self.kv_cache1[0]["k"].dtype == dtype
            and self.kv_cache1[0]["k"].device == device
        )

        if can_reuse_cache:
            for cache in self.kv_cache1:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["unrope_prev_k"] = None
                cache["unrope_prev_v"] = None
                cache["global_end_index"].zero_()
                cache["local_end_index"].zero_()
          
        else:
            self.kv_cache1 = None
            kv_cache1 = []

            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "unrope_prev_k": None,
                    "unrope_prev_v": None,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

            self.kv_cache1 = kv_cache1  # always store the clean cache
        return

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        can_reuse_cache = (
            self.crossattn_cache is not None
            and len(self.crossattn_cache) == self.num_transformer_blocks
            and self.crossattn_cache[0]["k"].shape[0] == batch_size
            and self.crossattn_cache[0]["k"].dtype == dtype
            and self.crossattn_cache[0]["k"].device == device
        )

        if can_reuse_cache:
            for cache in self.crossattn_cache:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["is_init"] = False
        else:
            self.crossattn_cache = None
            crossattn_cache = []

            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False
                })
            self.crossattn_cache = crossattn_cache
        return

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        profile: bool = False,
        low_memory: bool = False,
        encode_time: Optional[float] = None,
        is_prune: bool = False,
        prune_mask_info: Optional[dict] = None,
        is_unprune_kv: bool = False,
        from_memory: bool = False,
        trim_kv: Optional[List[int]] = None,
        track_mask: bool = False,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            is_prune (bool): if true, will prune the token for the input and unprune for the output.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling and all sorts of variables
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            prune_start = torch.cuda.Event(enable_timing=True)
            prune_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            
            init_start.record()

        total_time = 0.0

        # Step 1: Initialize KV cache to all zeros
        if trim_kv is not None:
            kv_cache_size = (trim_kv[0] + trim_kv[1]) * self.frame_seq_length
        else:
            kv_cache_size = 39000

        self._initialize_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                kv_cache_size=kv_cache_size
            )
        self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
        )

        # Step 2: Determine the prune mask
        prune_mask, compression_ratio = None, 1.0

        if is_prune:
            if profile: prune_start.record()

            prune_mask = rlt_prune(**prune_mask_info)
            compression_ratio = prune_mask[0].float().mean().item()
            
            if profile: prune_end.record()

        # Step 3: Cache context/ref feature for image to video generation
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            diffusion_start.record()

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        # Step 4: Denoising loop
        for current_num_frames in all_num_frames:

            if profile: block_start.record()
            frame_slice = slice(
                current_start_frame - num_input_frames,
                current_start_frame + current_num_frames - num_input_frames,
            )
            noisy_input = noise[:, frame_slice]
            
            if is_prune:
                mask_input = prune_mask[:, frame_slice]
            current_prune_mask = mask_input if is_prune else None
            
            # Step 4.1: Denoising time step loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        is_prune=is_prune,
                        is_unprune_kv=is_unprune_kv,
                        from_memory=from_memory,
                        trim_kv=trim_kv,
                        prune_mask=current_prune_mask,
                        update_prev=False,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    random_noise = torch.randn_like(denoised_pred)

                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        random_noise.flatten(0, 1),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        is_prune=is_prune,
                        is_unprune_kv=is_unprune_kv,
                        from_memory=from_memory,
                        trim_kv=trim_kv,
                        track_mask=track_mask,
                        prune_mask=current_prune_mask,
                        update_prev=True,
                    )

            # Step 4.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 4.3: rerun with timestep zero, for memory, to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise # context_noise = 0
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                is_prune=is_prune,
                is_unprune_kv=is_unprune_kv,
                from_memory=from_memory,
                trim_kv=trim_kv,
                prune_mask=current_prune_mask,
                update_prev=False,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 4.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            vae_start.record()

        # Step 5: Decode the output
        edit_video = self.vae.decode_to_pixel(output, use_cache=False)
        edit_video = (edit_video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            
            torch.cuda.synchronize()

            init_time = init_start.elapsed_time(init_end)
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            vae_time = vae_start.elapsed_time(vae_end)
            if is_prune: prune_time = prune_start.elapsed_time(prune_end)

            total_time = init_time + diffusion_time + vae_time + (encode_time if encode_time is not None else 0.0)
            denom = total_time if total_time > 0 else 1.0

            logger.info("Profiling results:")
            logger.info(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / denom:.2f}%)")
            logger.info(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / denom:.2f}%)")
            for i, block_time in enumerate(block_times):
                logger.info(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            logger.info(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / denom:.2f}%)")
            if encode_time is not None:
                logger.info(f"  - VAE encoding time: {encode_time:.2f} ms ({100 * encode_time / denom:.2f}%)")
            if is_prune: logger.info(f"  - Pruning time: {prune_time:.2f} ms ({100 * prune_time / denom:.2f}%)")
            
            logger.info(f"  - Total time: {total_time:.2f} ms")

        return edit_video, compression_ratio, total_time

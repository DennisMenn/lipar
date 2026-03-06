import argparse
import logging
import os
import torch
import torch.distributed as dist

from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from src.pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from src.utils.dataset import TextVideoPairDataset
from src.utils.misc import set_seed

from src.utils.demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from src.utils.tae import TAEHVDiffusersWrapper


logger = logging.getLogger("self_forcing.inference")


def get_video_filename(args, idx, seed_idx, prompt, model_tag, compression_ratio):
    if args.save_with_index:
        return f"{idx}-{seed_idx}_{model_tag}"
    
    prune_str = (f"{args.distance}_{args.patch_size}_{args.tau}-{args.tau2}--cr{compression_ratio:.3f}"
                if args.is_lip else "no_prune")
                
    parts = [
        prune_str,
        "_track" if args.track_mask else None,
        "tae" if args.use_tae else None,
        prompt[:100],
        str(seed_idx)
    ]
    return "--".join(p for p in parts if p)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use EMA parameters",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument(
        "--save_with_index",
        action="store_true",
        help="Whether to save the video using the index or prompt as the filename",
    )
    parser.add_argument(
        "--use_tae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use TAE (Tiny Auto Encoder)",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile throughput",
    )

    parser.add_argument(
        "--patch_size",
        type=str,
        default="1,2,2",
        help="Patch size for token pruning, format: temporal_dim,height,width",
    )
    parser.add_argument("--tau", type=float, default=0.149, help="Threshold for short-term difference in LIF pruning")
    parser.add_argument(
        "--tau2",
        type=float,
        default=0.3,
        help="Threshold for long-term difference in LIF pruning",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="L1_gaussian",
        choices=["L1", "L1_gaussian", "L2", "cos"],
        help="Distance metric for LIF pruning",
    )
    parser.add_argument(
        "--use_lipar",
        action="store_true",
        help="Apply default LIPAR settings: --is_lip --m_approx --n_aware_dup --m 1 6 (unless --m is set)",
    )
    parser.add_argument("--is_lip", action="store_true", help="Whether to perform Latent Inter-frame Pruning")
    parser.add_argument("--m_approx", action="store_true", help="Whether to perform m degree approximation for attention recovery")
    parser.add_argument(
        "--n_aware_dup",
        action="store_true",
        help="Whether to perform noise-aware duplication for attention recovery",
    )
    parser.add_argument(
        "--m",
        nargs=2,
        type=int,
        default=None,
        help="determine m for m-degree approximation; (1,6) is suggested (< first_frames, > last_frames)",
    )
    parser.add_argument("--track_mask", action="store_true", help="Whether to track where we prune the tokens")
    return parser.parse_args()


def apply_lipar_defaults(args):
    if args.use_lipar:
        args.is_lip = True
        args.m_approx = True
        args.n_aware_dup = True
        if args.m is None:
            args.m = [1, 6]


def validate_args(args):

    if args.m_approx and not args.is_lip:
        raise ValueError("--m_approx requires --is_lip")
    if args.n_aware_dup and not (args.is_lip and args.m_approx):
        raise ValueError("--n_aware_dup requires both --is_lip and --m_approx")
    if args.m is not None and not args.m_approx:
        raise ValueError("--m requires --m_approx")


def setup_distributed(seed):
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        set_seed(seed)
    return device, local_rank


def build_pipeline(config, args, device):
    if hasattr(config, "denoising_step_list"):
        if args.m is not None:
            config["model_kwargs"]["sink_size"] = args.m[0]
        if args.use_tae:
            return CausalInferencePipeline(config, device=device, vae=TAEHVDiffusersWrapper())
        return CausalInferencePipeline(config, device=device)

    if args.use_tae:
        raise NotImplementedError("TAE not implemented for multi-step inference yet")
    return CausalDiffusionInferencePipeline(config, device=device)


def create_dataloader(args):
    ops = [transforms.Resize((480, 832))]
    if not args.use_tae:
        ops.append(transforms.Normalize([0.5], [0.5]))
    transform = transforms.Compose(ops)

    dataset = TextVideoPairDataset(args.data_path, transform=transform, duplication=args.num_samples)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)
    return dataset, dataloader


def configure_logging(profile_enabled: bool, local_rank: int, output_folder: str):
    log_format = "%(message)s"
    logging.basicConfig(level=logging.WARNING, format=log_format, force=True)

    project_logger = logging.getLogger("self_forcing")
    project_logger.handlers.clear()
    project_logger.propagate = False

    if profile_enabled and local_rank == 0:
        project_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        project_logger.addHandler(stream_handler)

        log_dir = output_folder or "."
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "profile.log")
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        project_logger.addHandler(file_handler)

        logger.info(f"[Profile] Logging to: {log_file_path}")
    else:
        project_logger.setLevel(logging.WARNING)


def main():
    args = parse_args()
    apply_lipar_defaults(args)
    validate_args(args)

    device, local_rank = setup_distributed(args.seed)
    configure_logging(args.profile, local_rank, args.output_folder)

    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    pipeline = build_pipeline(config, args, device)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = "generator_ema" if args.use_ema else "generator"
        pipeline.generator.load_state_dict(state_dict[key])

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    dataset, dataloader = create_dataloader(args)
    if dist.is_initialized():
        dist.barrier()

    if local_rank == 0:
        print(f"Number of prompts: {len(dataset)}")
        os.makedirs(args.output_folder, exist_ok=True)

    total_time = 0.0
    total_frames = 0
    for _, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()
        batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

        prompt = batch["prompts"][0]
        item_name = prompt
        if hasattr(dataset, "video_path_list") and idx < len(dataset.video_path_list):
            video_path = dataset.video_path_list[idx]
            item_name = os.path.basename(video_path) if video_path else prompt

        if args.profile and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=gpu)

        src_video = batch["video"][0].to(device=device, dtype=torch.bfloat16)
        src_video = src_video[:, : src_video.shape[1] - src_video.shape[1] % 12, ...]
        if not args.use_tae:
            src_video = src_video.permute(0, 2, 1, 3, 4)

        encode_time = None
        if args.profile:
            encode_start = torch.cuda.Event(enable_timing=True)
            encode_end = torch.cuda.Event(enable_timing=True)
            encode_start.record()

        video_latent = pipeline.vae.encode_to_latent(src_video).to(device=device, dtype=torch.bfloat16)
        if args.profile:
            encode_end.record()
            torch.cuda.synchronize()
            encode_time = encode_start.elapsed_time(encode_end)

        time_step = pipeline.denoising_step_list[0] * torch.ones(
            [video_latent.shape[0] * video_latent.shape[1]], device=device, dtype=torch.long
        )
        random_noise = torch.randn_like(video_latent)
        sampled_noise = pipeline.scheduler.add_noise(
            video_latent.flatten(0, 1),
            random_noise.flatten(0, 1),
            time_step,
        ).unflatten(0, video_latent.shape[:2])

        prune_mask_info = {
            "x": video_latent,
            "distance": args.distance,
            "patch_dims": list(map(int, args.patch_size.split(","))),
            "drop_param": args.tau,
            "drop_param2": args.tau2,
        }

        edit_video, compression_ratio, elapsed_time = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            initial_latent=None,
            low_memory=low_memory,
            profile=args.profile,
            encode_time=encode_time,
            is_prune=args.is_lip,
            prune_mask_info=prune_mask_info if args.is_lip else None,
            is_unprune_kv=args.m_approx,
            from_memory=args.n_aware_dup,
            trim_kv=args.m,
            track_mask=args.track_mask,
        )

        edit_video = 255.0 * rearrange(edit_video, "b t c h w -> b t h w c").cpu()
        frame_count = int(edit_video.shape[1])
        total_frames += frame_count
        pipeline.vae.model.clear_cache()

        model_tag = "ema" if args.use_ema else "regular"
        sample_idx = int(idx / args.num_samples)
        seed_idx = idx % args.num_samples
        fname = get_video_filename(args, sample_idx, seed_idx, prompt, model_tag, compression_ratio)
        write_video(os.path.join(args.output_folder, f"{fname}.mp4"), edit_video[0], fps=16)
        total_time += elapsed_time

        if args.profile:
            denoise_time_sec = (elapsed_time / 1000.0) if elapsed_time is not None else 0.0
            fps = (frame_count / denoise_time_sec) if denoise_time_sec > 0 else 0.0

            per_video_peak_gpu_mem_mb = 0.0
           
        if torch.cuda.is_available():
            video_peak_gpu = torch.cuda.max_memory_allocated(device=gpu) / (1024 ** 3)
            if local_rank == 0:
                logger.info(f"[Profile] Item: {item_name}")
                logger.info(f"[Profile] Peak GPU Memory: {video_peak_gpu:.2f} GB")
                logger.info(
                    f"[Profile] FPS: {fps:.2f} frames/s "
                    f"({frame_count} frames / {denoise_time_sec:.3f} sec)"
                )

    if args.profile and local_rank == 0:
        total_time_sec = total_time / 1000.0 if total_time is not None else 0.0
        avg_fps = (total_frames / total_time_sec) if total_time_sec > 0 else 0.0
        logger.info(f"[Profile] Total Editing Time: {total_time:.2f} ms ({total_time_sec:.3f} sec)")
        logger.info(
            f"[Profile] Average FPS: {avg_fps:.2f} frames/s "
            f"({total_frames} total frames / {total_time_sec:.3f} sec)"
        )
    else:
        print(f"total editing time: {total_time}")


if __name__ == "__main__":
    main()
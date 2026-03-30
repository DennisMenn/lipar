"""
Demo for Streaming LIPAR.
"""

import os
import re
import random
import time
import base64
import argparse
import hashlib
import subprocess
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import traceback
from omegaconf import OmegaConf
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import queue
from threading import Thread, Event, Lock

from src.pipeline import CausalInferencePipeline
from src.utils.tae import TAEHVDiffusersWrapper
from src.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from torchvision import transforms as T
from src.utils.demo_utils.utils import generate_timestamp, build_debug_frames
from src.utils.demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from src.utils.segmentation_mask import subsample_and_join_masks, process_block_seg_masks, masks_to_latent

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--demo_config', type=str, default='./configs/demo_config.yaml',
                    help='Path to demo config YAML file')
cli_args = parser.parse_args()

args = OmegaConf.load(cli_args.demo_config)

# If use_lipar is false, disable all pruning-related settings
if not args.use_lipar:
    args.is_lip = False
    args.m_approx = False
    args.n_aware_dup = False
    args.m = None

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

_nvtx_depth = 0
def sync_if_profile(name=None):
    global _nvtx_depth
    if getattr(args, 'profile', False):
        if name is not None:
            if _nvtx_depth > 0:
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(name)
            _nvtx_depth = 1
        else:
            if _nvtx_depth > 0:
                torch.cuda.nvtx.range_pop()
                _nvtx_depth = 0

# Load models
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.m is not None:
    config["model_kwargs"]["sink_size"] = args.m[0]

text_encoder = WanTextEncoder()

# Global variables
current_vae_decoder = None
fp8_applied = False
frame_number = 0
anim_name = ""
frame_rate = 6
frame_buffers = []
debug_frame_buffers = []
pixels_per_block = 12  # default, updated when generation starts (num_frame_per_block * 4)

def initialize_vae_decoder():
    """Initialize VAE decoder (TAEHV)"""
    global current_vae_decoder
    current_vae_decoder = TAEHVDiffusersWrapper()
    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float32)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    print(f"✅ VAE decoder initialized with TAEHV")
    return current_vae_decoder

# Initialize VAE decoder
vae_decoder = initialize_vae_decoder()
transformer = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
transformer.load_state_dict(state_dict['generator_ema'])

text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.bfloat16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)
if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
transformer.to(gpu)

# Flask and SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'frontend_buffered_demo'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024)

generation_active = False
stop_event = Event()
pause_event = Event()
frame_send_queue = queue.Queue()
webcam_frame_cache = []
webcam_mask_cache = []
webcam_cache_lock = Lock()
webcam_block_ready = Event()
webcam_decode_time = 0.0
sender_thread = None
webcam_resize = T.Compose([T.Resize((480, 832))])
pending_prompt = None
pending_prompt_lock = Lock()


def tensor_to_base64_frame(frame_tensor):
    """Convert a single frame tensor to base64 image string."""
    global frame_number, frame_buffers
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()

    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[2] == 3:
        image = Image.fromarray(frame, 'RGB')
    else:
        image = Image.fromarray(frame)

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    frame_number += 1
    frame_buffers.append(buffer.getvalue())
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def frame_sender_worker():
    """Background thread that processes frame send queue non-blocking."""
    global frame_send_queue, generation_active, stop_event

    print("📡 Frame sender thread started")

    while True:
        frame_data = None
        try:
            frame_data = frame_send_queue.get(timeout=1.0)

            if frame_data is None:
                frame_send_queue.task_done()
                break

            frame_tensor, frame_index, block_index, job_id, copy_event = frame_data

            if copy_event is not None:
                copy_event.synchronize()

            base64_frame = tensor_to_base64_frame(frame_tensor)

            try:
                socketio.emit('frame_ready', {
                    'data': base64_frame,
                    'frame_index': frame_index,
                    'block_index': block_index,
                    'job_id': job_id
                })
            except Exception as e:
                print(f"⚠️ Failed to send frame {frame_index}: {e}")

            frame_send_queue.task_done()

        except queue.Empty:
            if not generation_active and frame_send_queue.empty():
                break
        except Exception as e:
            print(f"❌ Frame sender error: {e}")
            if frame_data is not None:
                try:
                    frame_send_queue.task_done()
                except Exception as e:
                    print(f"❌ Failed to mark frame task as done: {e}")
            break

    print("📡 Frame sender thread stopped")

@torch.no_grad()
def generate_video_stream(prompt, seed, enable_fp8=False):
    """Generate video and push frames immediately to frontend."""
    global generation_active, stop_event, frame_send_queue, sender_thread, fp8_applied, current_vae_decoder, frame_rate, anim_name, pixels_per_block, frame_buffers, debug_frame_buffers

    total_denoising_time = 0.0
    total_decode_time = 0.0

    try:
        generation_active = True
        stop_event.clear()
        pause_event.clear()
        job_id = generate_timestamp()

        # Start frame sender thread if not already running
        if sender_thread is None or not sender_thread.is_alive():
            sender_thread = Thread(target=frame_sender_worker, daemon=True)
            sender_thread.start()

        def emit_progress(message, progress):
            try:
                socketio.emit('progress', {
                    'message': message,
                    'progress': progress,
                    'job_id': job_id
                })
            except Exception as e:
                print(f"❌ Failed to emit progress: {e}")

        emit_progress('Starting generation...', 0)

        # Handle FP8 quantization
        if enable_fp8 and not fp8_applied:
            try:
                emit_progress('Applying FP8 quantization...', 3)
                print("🔧 Applying FP8 quantization to transformer")
                from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
                quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
                fp8_applied = True
                print("✅ FP8 quantization applied successfully")
            except Exception as e:
                error_msg = f"FP8 quantization failed: {str(e)}"
                print(f"⚠️ {error_msg}")
                socketio.emit('show_warning', {
                    'message': f'<strong>⚠️ FP8 failed:</strong> {str(e)}<br><small>Continuing without FP8 quantization...</small>'
                })

        # Text encoding
        emit_progress('Encoding text prompt...', 8)
        conditional_dict = text_encoder(text_prompts=[prompt])
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.bfloat16)
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Initialize generation
        emit_progress('Initializing generation...', 12)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rnd = torch.Generator(gpu).manual_seed(seed)
        trim_kv = list(args.m) if args.m is not None else None
        if trim_kv is not None:
            kv_cache_size = (trim_kv[0] + trim_kv[1]) * pipeline.frame_seq_length
        else:
            kv_cache_size = 39000

        pipeline._initialize_cache(batch_size=1, dtype=torch.bfloat16, device=gpu, kv_cache_size=kv_cache_size)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=gpu)
        transformer.flow_prev = None
        transformer.noisy_prev = None

        # Streaming encoding setup
        pixels_per_block = pipeline.num_frame_per_block * 4
        enc_mem = None
        prev_seg_mask = None
        prev_prev_seg_mask = None

        # Generation parameters
        current_start_frame = 0
        current_num_frames = pipeline.num_frame_per_block
        vae_cache = None

        total_frames_sent = 0
        generation_start_time = time.time()

        emit_progress('Waiting for webcam frames...', 15)

        # Clear stale cache from a previous run
        with webcam_cache_lock:
            webcam_frame_cache.clear()
            webcam_mask_cache.clear()
        webcam_block_ready.clear()

        # Signal frontend that we are ready to receive webcam frames
        socketio.emit('webcam_ready')

        idx = 0
        while generation_active and not stop_event.is_set():
            # Check for pause
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.1)
            if stop_event.is_set():
                break

            sync_if_profile("waiting")
            # Wait until we have enough frames
            t_request = time.time()
            while True:
                if not generation_active or stop_event.is_set():
                    break
                with webcam_cache_lock:
                    if len(webcam_frame_cache) >= pixels_per_block:
                        break
                webcam_block_ready.wait(timeout=0.05)
                webcam_block_ready.clear()
            t_received = time.time()
            wait_time = t_received - t_request

            if not generation_active or stop_event.is_set():
                break
            t_mask_preprocess = time.time()
            with webcam_cache_lock:
                all_frames = list(webcam_frame_cache)
                all_masks = list(webcam_mask_cache)
                webcam_frame_cache.clear()
                webcam_mask_cache.clear()

            if len(all_frames) == 0:
                continue

            # Check for mid-stream prompt change
            global pending_prompt
            force_full_mask = False
            with pending_prompt_lock:
                new_prompt = pending_prompt
                if new_prompt is not None:
                    pending_prompt = None
            if new_prompt is not None:
                print(f"🔄 Prompt changed mid-stream to: {new_prompt!r}")
                conditional_dict = text_encoder(text_prompts=[new_prompt])
                for key, value in conditional_dict.items():
                    conditional_dict[key] = value.to(dtype=torch.bfloat16)
                if low_memory:
                    gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
                    move_model_to_device_with_memory_preservation(
                        text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                pipeline._initialize_cache(batch_size=1, dtype=torch.bfloat16, device=gpu, kv_cache_size=kv_cache_size)
                pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=gpu)
                transformer.flow_prev = None
                transformer.noisy_prev = None
                current_start_frame = 0
                enc_mem = None
                prev_seg_mask = None
                prev_prev_seg_mask = None
                vae_cache = None
                force_full_mask = True
                idx = 0
                print("✅ Caches reset, restarting from block 0")

            sync_if_profile("preprocessing")

            num_received = len(all_frames)
            print(f"⏱️ Block {idx+1} [1.wait]: {wait_time:.3f}s | {num_received} frames received")

            # Subsample + stack
            block_pixel_frames, block_segmentation_masks = subsample_and_join_masks(all_frames, all_masks)
            pixel_block = torch.stack([f.to(device=gpu, non_blocking=True) for f in block_pixel_frames]).unsqueeze(0).float()
            mask_preprocess = time.time() - t_mask_preprocess

            # Encoding (VAE encode + noise)
            sync_if_profile("encoding")
            t_enc = time.time()
            block_latent, enc_mem = pipeline.vae.encode_to_latent(pixel_block, mem=enc_mem)
            block_latent = block_latent.to(device=gpu, dtype=torch.bfloat16)

            time_step = pipeline.denoising_step_list[0] * torch.ones(
                [1 * current_num_frames], device=gpu, dtype=torch.long)
            block_noise = torch.randn(block_latent.shape, device=gpu, dtype=torch.bfloat16, generator=rnd)
            noisy_input = pipeline.scheduler.add_noise(
                block_latent.flatten(0, 1), block_noise.flatten(0, 1), time_step
            ).unflatten(0, block_latent.shape[:2])

            sync_if_profile("prune_mask")
            enc_time = time.time() - t_enc
            print(f"⏱️ Block {idx+1} [2.encoding]: {enc_time:.3f}s")

            # Compute prune mask from segmentation masks
            t_mask = time.time()
            current_prune_mask = None
            if args.is_lip:
                if idx == 0 or force_full_mask:
                    current_prune_mask = torch.ones(block_latent.shape, dtype=torch.bool, device=gpu)
                else:
                    processed_masks, raw_last = process_block_seg_masks(block_segmentation_masks, prev_seg_mask, prev_prev_seg_mask, kernel_size=32, device=gpu)
                    prev_prev_seg_mask = block_segmentation_masks[1]
                    prev_seg_mask = block_segmentation_masks[2]
                    current_prune_mask = masks_to_latent(processed_masks, block_latent.shape, (1, 2, 2), gpu)

            mask_time = time.time() - t_mask
            print(f"⏱️ Block {idx+1} [3.prune_mask]: {mask_time+mask_preprocess:.3f}s")

            # Denoising loop + KV cache update
            sync_if_profile("denoising")
            t_denoise = time.time()

            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                if not generation_active or stop_event.is_set():
                    break

                timestep = torch.ones([1, current_num_frames], device=gpu,
                                      dtype=torch.int64) * current_timestep

                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length,
                        is_prune=args.is_lip,
                        is_unprune_kv=args.m_approx,
                        from_memory=args.n_aware_dup,
                        trim_kv=list(args.m) if args.m is not None else None,
                        prune_mask=current_prune_mask,
                        update_prev=False,
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn(denoised_pred.flatten(0, 1).shape, dtype=denoised_pred.dtype, device=denoised_pred.device, generator=rnd),
                        next_timestep * torch.ones([1 * current_num_frames], device=gpu, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    kv_prune_mask = current_prune_mask.clone()
                    current_prune_mask[:, 2] = True
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length,
                        is_prune=args.is_lip,
                        is_unprune_kv=args.m_approx,
                        from_memory=args.n_aware_dup,
                        trim_kv=list(args.m) if args.m is not None else None,
                        prune_mask=current_prune_mask,
                        update_prev=True,
                    )
                    current_prune_mask = kv_prune_mask

            if not generation_active or stop_event.is_set():
                break

            # KV cache update
            transformer(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=torch.zeros_like(timestep),
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
                is_prune=args.is_lip,
                is_unprune_kv=args.m_approx,
                from_memory=args.n_aware_dup,
                trim_kv=list(args.m) if args.m is not None else None,
                prune_mask=current_prune_mask,
                update_prev=False,
            )
            sync_if_profile("decoding")
            denoise_time = time.time() - t_denoise
            print(f"⏱️ Block {idx+1} [4.denoising]: {denoise_time:.3f}s")
            total_denoising_time += denoise_time

            # Decoding (VAE decode)
            t_decode = time.time()
            if vae_cache is not None:
                denoised_pred_decode = torch.cat([vae_cache, denoised_pred], dim=1)
            else:
                denoised_pred_decode = denoised_pred
            vae_cache = denoised_pred[:, -3:, :, :, :]
            pixels = current_vae_decoder.decode_to_pixel(denoised_pred_decode.float(), use_cache=False)
            if idx > 0:
                pixels = pixels[:, 12:, :, :, :]
            sync_if_profile("postprocess")
            decode_time = time.time() - t_decode
            print(f"⏱️ Block {idx+1} [5.decoding]: {decode_time:.3f}s")
            total_decode_time += decode_time

            # Postprocess (frame queue + debug)
            to_cpu_and_send(
                pixels, total_frames_sent, idx, job_id,
                block_pixel_frames, current_prune_mask,
                block_segmentation_masks, prev_seg_mask, prev_prev_seg_mask)

            block_frames = pixels.shape[1]
            total_frames_sent += block_frames
            block_total_time = time.time() - t_request
            compute_time = mask_preprocess + mask_time + enc_time + denoise_time + decode_time
            block_fps = block_frames / block_total_time if block_total_time > 0 else 0
            print(f"✅ Block {idx+1} total: {block_total_time:.2f}s (wait: {wait_time:.2f}s, compute: {compute_time:.2f}s, {block_frames} frames, {block_fps:.2f} fps)")

            # Send block fps to frontend so output playback matches generation rate
            try:
                socketio.emit('block_fps', {'fps': block_fps, 'block_index': idx})
            except Exception as e:
                print(f"⚠️ Failed to emit block_fps: {e}")

            current_start_frame += current_num_frames
            idx += 1

        generation_time = time.time() - generation_start_time
        print(f"🎉 Generation completed in {generation_time:.2f}s! {total_frames_sent} frames queued")
        if total_denoising_time > 0:
            print(f"🧮 Denoising: {total_denoising_time:.2f}s ({total_frames_sent/total_denoising_time:.2f} fps)")
        if total_decode_time > 0:
            print(f"🧮 Decoding: {total_decode_time:.2f}s ({total_frames_sent/total_decode_time:.2f} fps)")

        # Wait for all frames to be sent
        emit_progress('Sending remaining frames...', 97)
        frame_send_queue.join()

        # Save output video
        avg_fps = total_frames_sent / generation_time if generation_time > 0 else frame_rate
        generate_mp4_from_memory("./videos/"+anim_name+".mp4", avg_fps)
        if getattr(args, 'save_debug_video', False) and debug_frame_buffers:
            saved_buffers = frame_buffers
            frame_buffers = debug_frame_buffers
            generate_mp4_from_memory("./videos/"+anim_name+"_debug.mp4", avg_fps)
            frame_buffers = saved_buffers
            print(f"Debug video saved to ./videos/{anim_name}_debug.mp4")

        emit_progress('Generation complete!', 100)

        try:
            socketio.emit('generation_complete', {
                'message': 'Video generation completed!',
                'total_frames': total_frames_sent,
                'generation_time': f"{generation_time:.2f}s",
                'job_id': job_id
            })
        except Exception as e:
            print(f"❌ Failed to emit generation complete: {e}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        traceback.print_exc()
        try:
            socketio.emit('error', {
                'message': f'Generation failed: {str(e)}\n{traceback.format_exc()}',
                'job_id': job_id
            })
        except Exception as emit_err:
            print(f"❌ Failed to emit error: {emit_err}")
    finally:
        generation_active = False
        stop_event.set()
        try:
            frame_send_queue.put(None)
        except Exception as e:
            print(f"❌ Failed to put None in frame_send_queue: {e}")


def generate_mp4_from_memory(output_video_path, fps=24):
    """Generate an MP4 video by piping in-memory JPEG frames to ffmpeg."""
    global frame_buffers
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'image2pipe',
        '-framerate', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for buf in frame_buffers:
            proc.stdin.write(buf)
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            print(f"Video saved to {output_video_path}")
        else:
            print(f"ffmpeg exited with code {proc.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Streaming LIPAR demo'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('start_generation')
def handle_start_generation(data):
    global generation_active, frame_number, anim_name, frame_rate, frame_buffers, debug_frame_buffers

    frame_number = 0
    frame_buffers = []
    debug_frame_buffers = []
    if generation_active:
        emit('error', {'message': 'Generation already in progress'})
        return

    prompt = data.get('prompt', '')

    seed = getattr(args, 'seed', -1)
    if seed == -1:
        seed = random.randint(0, 2**32)

    words_up_to_punctuation = re.split(r'[^\w\s]', prompt)[0].strip() if prompt else ''
    if not words_up_to_punctuation:
        words_up_to_punctuation = re.split(r'[\n\r]', prompt)[0].strip()

    sha256_hash = calculate_sha256(prompt)
    anim_name = f"{words_up_to_punctuation[:20]}_{str(seed)}_{sha256_hash[:10]}"

    generation_active = True
    enable_fp8 = args.get('fp8', False)
    frame_rate = data.get('fps', 6)

    if not prompt:
        emit('error', {'message': 'Prompt is required'})
        return

    socketio.start_background_task(generate_video_stream, prompt, seed, enable_fp8)
    emit('status', {'message': 'Generation started'})


@socketio.on('webcam_block')
def handle_webcam_block(data):
    """Receive a batch of webcam frames with segmentation masks from the frontend."""
    global webcam_decode_time
    if not generation_active:
        return
    try:
        decode_start = time.time()
        frames = []
        masks = []
        for item in data:
            frame_data = item['frame']
            mask_data = item.get('mask')

            img_data = frame_data.split(',', 1)[1]
            img_bytes = base64.b64decode(img_data)
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            frame = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            frame = webcam_resize(frame)
            frames.append(frame)

            if mask_data:
                mask_img_data = mask_data.split(',', 1)[1]
                mask_bytes = base64.b64decode(mask_img_data)
                mask_image = Image.open(BytesIO(mask_bytes)).convert('L')
                mask_tensor = torch.tensor(np.array(mask_image), dtype=torch.float32) / 255.0
                mask_tensor = webcam_resize(mask_tensor.unsqueeze(0)).squeeze(0)
                masks.append(mask_tensor)
            else:
                masks.append(None)

        decode_elapsed = time.time() - decode_start
        with webcam_cache_lock:
            webcam_frame_cache.extend(frames)
            webcam_mask_cache.extend(masks)
            webcam_decode_time = decode_elapsed
            webcam_block_ready.set()
    except Exception as e:
        print(f"⚠️ Failed to process webcam block: {e}")
        traceback.print_exc()


@socketio.on('pause_generation')
def handle_pause_generation():
    global pause_event
    pause_event.set()
    with webcam_cache_lock:
        webcam_frame_cache.clear()
        webcam_mask_cache.clear()
    print("⏸️ Generation paused")

@socketio.on('resume_generation')
def handle_resume_generation():
    global pause_event
    pause_event.clear()
    print("▶️ Generation resumed")
    with webcam_cache_lock:
        webcam_frame_cache.clear()
        webcam_mask_cache.clear()
    webcam_block_ready.clear()
    emit('status', {'message': 'Generation resumed'})


@socketio.on('update_prompt')
def handle_update_prompt(data):
    global pending_prompt
    new_prompt = data.get('prompt', '')
    if not new_prompt:
        emit('error', {'message': 'Prompt is required'})
        return
    with pending_prompt_lock:
        pending_prompt = new_prompt
    print(f"📝 Received prompt update: {new_prompt!r}")
    emit('status', {'message': f'Prompt will update on next block'})


copy_stream = torch.cuda.Stream()

def to_cpu_and_send(pixels, total_frames_sent, idx, job_id,
                    block_pixel_frames=None, current_prune_mask=None,
                    block_segmentation_masks=None, prev_seg_mask=None, prev_prev_seg_mask=None):
    pixels_cpu_batch = torch.empty(pixels[0].shape, dtype=torch.float32, pin_memory=True)

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(torch.cuda.default_stream())
        pixels_cpu_batch.copy_(pixels[0], non_blocking=True)
        pixels[0].record_stream(copy_stream)
        copy_event = torch.cuda.Event()
        copy_event.record(copy_stream)

    for frame_idx in range(pixels_cpu_batch.shape[0]):
        if not generation_active or stop_event.is_set():
            break
        frame_send_queue.put((pixels_cpu_batch[frame_idx], total_frames_sent + frame_idx, idx, job_id, copy_event))

    if getattr(args, 'save_debug_video', False):
        debug_frame_buffers.extend(
            build_debug_frames(block_pixel_frames, current_prune_mask, pixels,
                                block_segmentation_masks, prev_seg_mask, prev_prev_seg_mask))

@app.route('/')
def index():
    return render_template('demo.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'generation_active': generation_active,
        'free_vram_gb': get_cuda_free_memory_gb(gpu),
    })


if __name__ == '__main__':
    print(f"🚀 Starting demo on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)

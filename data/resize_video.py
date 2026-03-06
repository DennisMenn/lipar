import json
import os
import math
from pathlib import Path
import cv2
import argparse
from pdb import set_trace

TARGET_H = 480
TARGET_W = 832

def load_video_paths_from_dir(dir_path: Path):
    """
    REPLACED: Now scans a directory for all .mp4 files 
    instead of parsing an NDJSON file.
    """
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a valid directory.")
        return []
    
    # Finds all .mp4 files in the directory
    resolved_paths = list(dir_path.glob("*.mp4"))
    return resolved_paths

def ensure_output_dir(out_path: Path):
    """Create output directory if it doesn't exist"""
    if not out_path.exists():
        print(f"Creating output directory: {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
    return out_path

def resize_video(src: Path, dst_dir: Path, target_fps: int, max_frames: int):
    """
    Logic remains UNCHANGED from your original code.
    """
    if not src.exists():
        print(f"Skip (not found): {src}")
        return
    
    # Create output path (keeping original filename)
    out_path = dst_dir / src.name
    
    # Open video file
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Failed to open: {src}")
        return
    
    # Get video properties
    src_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # --- Logic to determine target frame count (Multiple of 12) ---
    frames_to_keep = None
    target_count = 0
    
    # 1. Determine the raw limit based on max_frames
    # We floor max_frames to the nearest multiple of 12 (e.g., 100 -> 96)
    limit_multiple_12 = (max_frames // 12) * 12
    
    if src_total_frames > 0:
        if src_total_frames > max_frames:
            # Scenario: Downsample
            # Target is the max limit (multiple of 12)
            target_count = limit_multiple_12
            
            # Create set of indices to keep
            step = src_total_frames / target_count
            frames_to_keep = set([int(i * step) for i in range(target_count)])
            print(f"Downsampling {src.name}: {src_total_frames} -> {target_count} frames (Multiple of 12)")
        else:
            # Scenario: Truncate / Keep
            # Target is the current total floored to multiple of 12
            target_count = (src_total_frames // 12) * 12
            # No specific indices needed, we just stop writing after target_count
            frames_to_keep = None
            print(f"Truncating {src.name}: {src_total_frames} -> {target_count} frames (Multiple of 12)")
    else:
        # Fallback if source frame count is unknown
        target_count = limit_multiple_12
        frames_to_keep = None

    if target_count == 0:
        print(f"Skipping {src.name}: resulting frame count would be 0 (Source < 12 frames?)")
        cap.release()
        return

    # Choose codec based on output format
    if src.suffix.lower() in ('.mp4', '.m4v', '.mov'):
        # CHANGE THIS LINE FROM 'mp4v' TO 'avc1'
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create video writer with the TARGET FPS
    writer = cv2.VideoWriter(str(out_path), fourcc, target_fps, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        print(f"Failed to create writer for: {out_path}")
        cap.release()
        return
    
    # Process frames
    read_frame_idx = 0
    written_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Logic to decide if we write this frame
        should_write = False
        
        # Stop early if we have written enough frames (for the non-downsampling case)
        if written_frame_idx >= target_count:
            break

        if frames_to_keep is None:
            # Sequential writing
            should_write = True
        else:
            # Downsampling specific indices
            if read_frame_idx in frames_to_keep:
                should_write = True

        if should_write:
            # Resize frame
            resized_frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            writer.write(resized_frame)
            written_frame_idx += 1

        read_frame_idx += 1
        
        if read_frame_idx % 100 == 0:
            print(f"{src.name}: processed source frame {read_frame_idx}/{src_total_frames if src_total_frames>0 else '?'}")

    # Clean up
    cap.release()
    writer.release()
    print(f"Done: {out_path} (Wrote {written_frame_idx} frames at {target_fps} fps)")

def parse_args():
    parser = argparse.ArgumentParser(description="Resize videos in a directory to fixed resolution with frame control.")
    # Changed from json_path to input_dir
    parser.add_argument("input_dir", type=Path, help="Directory containing input .mp4 files.")
    parser.add_argument("output", type=Path, help="Output directory for resized videos.")
    
    parser.add_argument("--fps", type=int, default=16, help="Target output frame rate (default: 16).")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames per video. Inputs larger than this will be downsampled (default: 100).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir # Changed
    output_dir = args.output
    target_fps = args.fps
    max_frames = args.max_frames
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # Ensure output directory exists
    dst_dir = ensure_output_dir(output_dir)
    
    # Load video paths from directory instead of JSON
    videos = load_video_paths_from_dir(input_dir)
    
    if not videos:
        print(f"No .mp4 files found in {input_dir}.")
        return
    
    print(f"Found {len(videos)} video(s). Output dir: {dst_dir}")
    print(f"Settings -> FPS: {target_fps}, Max Frames: {max_frames} (Will adjust to nearest multiple of 12)")
    
    # Process each video
    for video_path in videos:
        resize_video(video_path, dst_dir, target_fps, max_frames)

if __name__ == "__main__":
    main()
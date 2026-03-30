"""
Pixel-space motion mask using MOG2.

Functions:
    compute_pixel_mask: Compute mask for 4 frames (quarter-res).
    compute_pixel_mask_12: Compute 3 downsampled masks for 12 frames.

Standalone usage: python pixel_mask.py --video <path> [--out mask_output.mp4]
"""
import cv2
import numpy as np

DOWNSCALE = 4
BLOCK_SIZE = 4
MORPH_KERNEL = np.ones((5, 5), np.uint8)
SOBEL_ZONE_KERNEL = np.ones((13, 13), np.uint8)
GAUSSIAN_KSIZE = (5, 5)
MIN_CONTOUR_AREA = 625
SOBEL_THRESHOLD = 50


def compute_pixel_mask(frames, mog2):
    """Compute pixel mask for 4 input frames at quarter resolution.

    Args:
        frames: list of 4 BGR frames (already downscaled).
        mog2: cv2.BackgroundSubtractorMOG2 instance (not updated).

    Returns:
        fg: binary mask (uint8, 0/255).
    """
    fg_masks = []
    for frame in frames:
        raw = cv2.GaussianBlur(frame, GAUSSIAN_KSIZE, 0)
        fg = mog2.apply(raw, learningRate=0)
        fg_masks.append(fg)
    fg = np.bitwise_or.reduce(fg_masks)

    last_frame = frames[-1]
    gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))
    sobel_mask = (sobel > SOBEL_THRESHOLD).astype(np.uint8) * 255

    num_labels, labels = cv2.connectedComponents(sobel_mask)
    connection_pts = cv2.bitwise_and(fg, sobel_mask)
    connected_labels = set(np.unique(labels[connection_pts > 0])) - {0}
    connected_mask = np.isin(labels, list(connected_labels)).astype(np.uint8) * 255
    zone = cv2.dilate(connection_pts, SOBEL_ZONE_KERNEL)
    fg = cv2.bitwise_or(fg, cv2.bitwise_and(connected_mask, zone))

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, contours, -1, 255, cv2.FILLED)
    fg = cv2.dilate(fg, MORPH_KERNEL, iterations=2)

    fg[-1, :] = 255
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fg, contours, -1, 255, cv2.FILLED)
    fg[-1, :] = 0

    return fg


def compute_pixel_mask_12(frames, mog2, var_threshold=10):
    """Compute 3 downsampled binary masks from 12 frames.

    Downscales input by DOWNSCALE, computes masks for 3 non-overlapping
    groups of 4, then block-downsamples each mask by BLOCK_SIZE.
    Effective spatial downscale: DOWNSCALE * BLOCK_SIZE = 16.

    Args:
        frames: list of 12 BGR frames (full resolution).
        mog2: cv2.BackgroundSubtractorMOG2 instance (not updated).
        var_threshold: MOG2 variance threshold.

    Returns:
        fgs_ds: list of 3 binary masks (uint8, 0/1), shape (H/16, W/16).
    """
    assert len(frames) == 12, f"Expected 12 frames, got {len(frames)}"
    frames_ds = [cv2.resize(f, (f.shape[1] // DOWNSCALE, f.shape[0] // DOWNSCALE)) for f in frames]
    fgs = []
    for i in range(3):
        group = frames_ds[i * 4 : (i + 1) * 4]
        fg = compute_pixel_mask(group, mog2)
        fgs.append(fg)
    h, w = fgs[0].shape
    h_ds, w_ds = h // BLOCK_SIZE, w // BLOCK_SIZE
    fgs_ds = []
    for fg in fgs:
        blocks = fg[:h_ds * BLOCK_SIZE, :w_ds * BLOCK_SIZE].reshape(h_ds, BLOCK_SIZE, w_ds, BLOCK_SIZE)
        fgs_ds.append((blocks.max(axis=(1, 3)) > 0).astype(np.uint8))
    return fgs_ds


def update_mog2(frames, mog2):
    """Update MOG2 model with frames (downscaled)."""
    for f in frames:
        f_ds = cv2.resize(f, (f.shape[1] // DOWNSCALE, f.shape[0] // DOWNSCALE))
        raw = cv2.GaussianBlur(f_ds, GAUSSIAN_KSIZE, 0)
        mog2.apply(raw)


if __name__ == "__main__":
    import argparse
    import time
    from collections import deque

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="mask_output.mp4")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'avc1'), fps, (w * 2, h))
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=False)

    fg_history = deque(maxlen=12)
    n = 0
    t_total = 0
    effective_scale = DOWNSCALE * BLOCK_SIZE

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_history.append(frame)
        if len(fg_history) == 12:
            t0 = time.time()

            fgs_ds = compute_pixel_mask_12(list(fg_history), mog2)

            for i in range(3):
                fg_up = fgs_ds[i].repeat(effective_scale, axis=0).repeat(effective_scale, axis=1) * 255
                overlay = np.zeros_like(fg_history[0])
                overlay[fg_up > 0] = [0, 255, 0]
                for j in range(4):
                    writer.write(np.hstack([fg_history[i * 4 + j], overlay]))

            update_mog2(list(fg_history), mog2)

            dt = time.time() - t0
            t_total += dt
            n += 12
            pcts = [(m > 0).sum() / m.size * 100 for m in fgs_ds]
            print(f"Frame {n-12:4d}-{n:4d} | mask: {pcts[0]:5.1f}% {pcts[1]:5.1f}% {pcts[2]:5.1f}% | {dt*1000:.1f}ms")

            fg_history.clear()

    cap.release()
    writer.release()
    print(f"\n{n} frames | avg {t_total/n*1000:.1f}ms/frame | {n/t_total:.1f} fps")
    print(f"Saved to {args.out}")

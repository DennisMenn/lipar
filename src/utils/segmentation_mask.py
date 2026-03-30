"""
Subsample accumulated frames/masks to 12 and join masks into 3 groups.

Functions:
    subsample_and_join_masks: Subsample N frames to 12, join masks into 3 groups.
    process_block_seg_masks: Join with prev, rolling OR, spatial dilation.
    masks_to_latent: Downsample masks to latent resolution via block-max.
"""
import torch
import torch.nn.functional as F


def subsample_and_join_masks(frames, masks):
    """Subsample N frames to 12 and join masks into 3 groups of 4.

    Args:
        frames: list of N tensors (N >= 12).
        masks: list of N masks (torch tensors), same length as frames.
            Each mask is a 2D tensor (H, W) with values in [0, 1].

    Returns:
        frames_12: list of 12 evenly-spaced frames.
        masks_3: list of 3 joined masks (element-wise max over groups of 4).
    """
    n = len(frames)
    assert n >= 12, f"Expected >= 12 frames, got {n}"
    assert len(masks) == n, f"frames ({n}) and masks ({len(masks)}) length mismatch"

    # Evenly spaced indices
    indices = torch.round(torch.linspace(0, n - 1, 12)).long()
    frames_12 = [frames[i] for i in indices]
    masks_12 = [masks[i] for i in indices]

    # Join masks in groups of 4 via element-wise max
    masks_3 = []
    for g in range(3):
        group = masks_12[g * 4 : (g + 1) * 4]
        joined = torch.stack(group).amax(dim=0)
        masks_3.append(joined)

    return frames_12, masks_3


def process_block_seg_masks(block_seg_masks, prev_seg_mask, prev_prev_seg_mask, kernel_size=3, device='cuda'):
    """Join with prev masks, rolling OR along temporal dim, then spatial dilation.

    Step 1 - Join with prev: joined[i] = max(block_seg_masks[i], prev_seg_mask, prev_prev_seg_mask)
    Step 2 - Rolling OR: out[0] = joined[0], out[i] = max(joined[i], out[i-1])
    Step 3 - Spatial dilation via max_pool2d on each (H, W) mask

    Saves raw block_seg_masks[2] (no join, no OR, no dilation) as prev for next block.

    Args:
        block_seg_masks: list of 3 masks (H, W), torch tensors with values in [0, 1].
        prev_seg_mask: previous block's last raw mask (H, W), or None.
        prev_prev_seg_mask: the block before prev's last raw mask (H, W), or None.
        kernel_size: spatial dilation kernel size.
        device: torch device to run on (default 'cuda').

    Returns:
        (processed_masks_3, raw_last): list of 3 processed masks (H, W) on device,
            and raw block_seg_masks[2] to save as next prev_seg_mask.
    """
    raw_last = block_seg_masks[2]

    # Move masks to GPU
    block_seg_masks = [m.to(device) for m in block_seg_masks]
    if prev_seg_mask is not None:
        prev_seg_mask = prev_seg_mask.to(device)
    if prev_prev_seg_mask is not None:
        prev_prev_seg_mask = prev_prev_seg_mask.to(device)

    # Step 1: Join with prev and prev_prev
    prevs = [p for p in [prev_seg_mask, prev_prev_seg_mask] if p is not None]
    if not prevs:
        joined = [torch.ones_like(block_seg_masks[0])] * 3
    else:
        joined = []
        for m in block_seg_masks:
            j = m
            for p in prevs:
                j = torch.maximum(j, p)
            joined.append(j)

    # Step 2: Rolling OR
    out = [joined[0]]
    out.append(torch.maximum(joined[1], out[0]))
    out.append(torch.maximum(joined[2], out[1]))

    # Step 3: Spatial dilation
    if kernel_size > 1:
        pad = kernel_size // 2
        dilated = []
        for m in out:
            m4d = m.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
            d = F.max_pool2d(m4d, kernel_size=kernel_size, stride=1, padding=pad)
            dilated.append(d.squeeze(0).squeeze(0))
        out = dilated

    return out, raw_last


def masks_to_latent(masks_3, block_latent_shape, patch_size, device):
    """Downsample 3 pixel-res masks to latent resolution via block-max.

    Matches rlt_prune output format: downsample to (H_lat/p1, W_lat/p2),
    then repeat_interleave by patch_size to get (H_lat, W_lat) with
    uniform p1×p2 blocks.

    Args:
        masks_3: list of 3 masks (H, W) torch tensors with values in [0, 1].
            H, W are pixel mask resolution (e.g. 480, 832).
        block_latent_shape: shape of block_latent (1, T, C, H_lat, W_lat).
            e.g. (1, 3, 16, 60, 104).
        patch_size: tuple (p0, p1, p2), e.g. (1, 2, 2).
        device: torch device.

    Returns:
        prune_mask: bool tensor (1, T, C, H_lat, W_lat).
    """
    _, T, C, H_lat, W_lat = block_latent_shape
    p1, p2 = patch_size[1], patch_size[2]
    H_patch, W_patch = H_lat // p1, W_lat // p2  # e.g. 30, 52
    H, W = masks_3[0].shape

    block_h = H // H_patch
    block_w = W // W_patch

    latent_masks = []
    for mask in masks_3:
        # Crop to exact multiple of block size
        cropped = mask[:H_patch * block_h, :W_patch * block_w]
        # Reshape into blocks and take max → (H_patch, W_patch)
        blocks = cropped.reshape(H_patch, block_h, W_patch, block_w)
        ds = blocks.amax(dim=(1, 3)) > 0
        latent_masks.append(ds)

    # Stack to (T, H_patch, W_patch)
    stacked = torch.stack(latent_masks).to(device)
    # Expand to (1, C, T, H_patch, W_patch) then repeat_interleave to match rlt_prune
    stacked = stacked.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H_patch, W_patch)
    stacked = stacked.expand(-1, C, -1, -1, -1)  # (1, C, T, H_patch, W_patch)
    stacked = stacked.repeat_interleave(p1, dim=3).repeat_interleave(p2, dim=4)  # (1, C, T, H_lat, W_lat)
    stacked = stacked.transpose(1, 2)  # (1, T, C, H_lat, W_lat)
    return stacked



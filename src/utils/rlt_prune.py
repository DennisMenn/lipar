import torch
import numpy as np
import cv2
from torch import Tensor
from typing import Tuple
from scipy.ndimage import median_filter, binary_dilation
import torch.nn.functional as F

from einops import rearrange
from torchvision.transforms import GaussianBlur

from pdb import set_trace

'''
Utilities to improve prune masks
'''
def cal_diffs(x1: torch.Tensor, x2: torch.Tensor, distance: str, patch_size=2) -> torch.Tensor:
    '''
    Calculate the difference between two tensors.
    Args:
        x1: Tensor of shape (B, C, T-1, H, W)
        x2: Tensor of shape (B, C, T-1, H, W)
    Returns:
        diff: Tensor of shape (B, C, T-1, H, W)
    '''
    def transform(x, patch_size):
        x = x.view(B, C, T, H//patch_size, patch_size, W//patch_size, patch_size) # (B, C, T, H/2, 2, W/2, 2)
        x = x.permute(0, 2, 3, 5, 4, 6, 1) # (B, T, H/2, W/2, 2, 2, C)
        x = x.contiguous().view(B, T, (H//patch_size), (W//patch_size), patch_size*patch_size*C)   #(B, T, H/2 * W/2, 2*2*C)  
        return x
    
    def sel_quantiles(x, lower_quantile=0, upper_quantile=1):
        sorted_x, _ = torch.sort(x.abs(), dim=-1)
        start_idx = int(lower_quantile * x.shape[-1])
        end_idx = int(upper_quantile * x.shape[-1])
        return sorted_x[:,:,:,:,start_idx:end_idx]

    B, C, T, H, W = x1.shape

    x1 = transform(x1, patch_size)
    x2 = transform(x2, patch_size)

    # Ensure we track negative movement.
    if distance == "L1" or distance == "L1_gaussian":
        diffs = x1-x2
        diffs = sel_quantiles(diffs, 0.2, 0.8)                               # [1, 20, 30, 52, 32]
        diffs = torch.norm(diffs, p=1, dim=-1) / diffs.shape[-1]   # [1, 20, 30, 52]

    elif distance == "L2":
        diffs = sel_quantiles(x1 - x2)
        diffs = torch.norm(diffs, p=2, dim=-1) / diffs.shape[-1] 
    elif distance == "cos":
        diffs = 1 - F.cosine_similarity(x1, x2, dim=-1)  #(B, T, H/2, W/2)

    diffs = diffs.reshape(B, 1, T, H//patch_size, W//patch_size)   #(B, 1, T, H/P, W/P)
    return diffs

def median_blur(diffs: torch.Tensor, size=(3,3,3), causal=True) -> torch.Tensor:
        '''
        input:  B=1 C=1 T H W
        output: B=1 C=1 T H W

        '''
        device = diffs.device
        diffs = diffs.squeeze()  # First channel: [T, H/P, W/P]
        diffs_np = diffs.cpu().numpy()
        if causal and size[0] > 1:
            # Pad temporal dim: (size[0]-1) on left, 0 on right
            pad_t = size[0] - 1
            diffs_np = np.pad(diffs_np, ((pad_t, 0), (0, 0), (0, 0)), mode='edge')
            result = median_filter(diffs_np, mode='nearest', size=size)
            result = result[pad_t:]  # remove temporal padding
        else:
            result = median_filter(diffs_np, mode='nearest', size=size)
        result = torch.from_numpy(result).to(device)
        return result.unsqueeze(0).unsqueeze(0)  # Output: [1, 1, T, H/P, H/P]

def morphology_operation(prune_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    '''
    input:  B=1 C=1 T H W
    output: B=1 C=1 T H W

    '''
    _, T, _, _ = prune_mask.shape[1:]

    device = prune_mask.device
    prune_mask = prune_mask.squeeze()  # First channel: [T, H, W]
    result = torch.empty_like(prune_mask)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for t in range(T): 
        mask = prune_mask[t].cpu().numpy().astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        result[t] = torch.from_numpy(mask > 0)

    result = result.to(device).unsqueeze(0).unsqueeze(0)
    return result 

def dilation(mask: torch.Tensor, structure=(3,5,5), causal=True) -> torch.Tensor:
    '''
    input:  B=1 C=1 T H W
    output: B=1 C=1 T H W
    '''
    device = mask.device
    mask = mask.squeeze()  # First channel: [T, H/P, W/P]
    struct = np.ones(structure)
    if causal and len(structure) == 3 and structure[0] > 1:
        # Zero out future timesteps (center is at structure[0]//2, zero everything after)
        struct[structure[0]//2 + 1:, :, :] = 0
    result = binary_dilation(mask.cpu().numpy(), structure=struct)

    result = torch.from_numpy(result).to(device)
    return result.unsqueeze(0).unsqueeze(0)  # Output: [1, 1, T, H/P, H/P]

def enforce_min_true_ratio(prune_mask: torch.Tensor, min_ratio: float, dilation_structure=(1, 3, 3)) -> torch.Tensor:
    '''
    Ensures at least min_ratio of tokens in prune_mask are True by iteratively dilating.
    If prune_mask is all False, the constraint is skipped.

    input:  B=1 C=1 T H W
    output: B=1 C=1 T H W
    '''
    T = prune_mask.shape[2]
    tokens_per_frame = prune_mask[:, :, 0].numel()
    spatial_structure = (dilation_structure[1], dilation_structure[2])

    for t in range(T):
        frame = prune_mask[:, :, t:t+1]  # (1, 1, 1, H, W)
        if not frame.any():
            continue
        while frame.sum().item() / tokens_per_frame < min_ratio:
            frame = dilation(frame, structure=spatial_structure)
        prune_mask[:, :, t:t+1] = frame

    return prune_mask

def get_gaussian_kernel3d(kernel_size, sigma, is_causal=False, device='cuda'):
    if isinstance(kernel_size, int):
        kt = kc = kh = kernel_size
    else:
        kt, kc, kh = kernel_size
    if isinstance(sigma, (int, float)):
        st = sc = sh = float(sigma)
    else:
        st, sc, sh = sigma
    
    t = torch.linspace(-(kt // 2), kt // 2, kt)
    c = torch.linspace(-(kc // 2), kc // 2, kc)
    h = torch.linspace(-(kh // 2), kh // 2, kh)
    
    grid_t, grid_c, grid_h = torch.meshgrid(t, c, h, indexing='ij')
    kernel = torch.exp(-(grid_t**2 / (2 * st**2) + grid_c**2 / (2 * sc**2) + grid_h**2 / (2 * sh**2))).to(device)
    if is_causal:
        kernel[kt//2+1:,:,:] = 0.0
    kernel = kernel / kernel.sum()

    return kernel

def gaussian_blur_3D(x, kernel_size=3, sigma=1.0, is_causal=False):
    '''
    input:  B=1 C=1 T H W
    output: B=1 C=1 T H W
    '''
    
    assert x.shape[0] and x.shape[1] == 1, "Input x must have shape (B=1, C=1, T, H, W)"
    
    if isinstance(kernel_size, int):
        kt = kh = kw = kernel_size
    else:
        kt, kh, kw = kernel_size
    
    pad_t = (kt - 1) // 2
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    padded_x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, pad_t), mode='replicate')
    
    kernel = get_gaussian_kernel3d(kernel_size, sigma, is_causal=is_causal, device=x.device)  # (kc, kt, kh)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kc, kt, kh)
    x = torch.nn.functional.conv3d(padded_x, kernel)
    return x

def bound_diff(prune_mask, x, distance, threshold, patch_size=2) -> torch.Tensor:
    '''
    Updates prune_mask based on the difference between memory token and denoising tokens 

    Args:
    prune_mask: shape (B=1, C=1, T, H, W)
    x: patchified video latent, shape (B=1, C, T, 2*H, 2*W)
    threshold: Difference to set mask true
    patch_size Patch size for diffs_cal 
    
    Returns:
        updated_mask: shape (B=1, C=1, T, H, W)
    '''
    # The first frame is always kept
    first_frame = torch.full_like(prune_mask[:, :, 0:1], bool(True))
    prune_mask = torch.cat([first_frame, prune_mask], dim=2)
    
    B, C, T, H, W = prune_mask.shape
    
    # Find the last true among the first 3 frames
    last_true_idx = 2 - torch.argmax(prune_mask[:, :, :3].int().flip(2), dim=2)  # Shape: (B=1, C=1, H, W)
    last_true_idx = last_true_idx.unsqueeze(2) \
                    .expand(B, x.shape[1], -1, -1, -1) \
                    .repeat_interleave(patch_size, dim=3) \
                    .repeat_interleave(patch_size, dim=4)
    
    prev_x = torch.gather(x[:, :, :3, :, :], dim=2, index=last_true_idx).squeeze(2).clone()
    temp_prev = prev_x.clone()
      
    for t in range(3,T):
        diffs = cal_diffs(prev_x.unsqueeze(2), x[:, :, t].unsqueeze(2), distance, patch_size).squeeze(2) 
        prune_mask[:, :, t] = (diffs > threshold) | prune_mask[:, :, t]

        exp_prune_mask = prune_mask[:, :, t].expand(-1, x.shape[1], -1, -1) \
                            .repeat_interleave(patch_size, dim=2) \
                            .repeat_interleave(patch_size, dim=3)
        temp_prev[exp_prune_mask] = x[:, :, t][exp_prune_mask].clone()
        if t % 3 == 2: # last frame of 3
            prev_x = temp_prev.clone()
 
    return prune_mask[:, :, 1:]


'''
Functions for pruning and unpruning

'''
def rlt_prune(x: Tensor, distance: str, patch_dims: Tuple[int, int, int] = (1, 2, 2), drop_param: float = 0.1, drop_param2 = None, min_true_ratio: float = None, state: dict = None):
    '''
    x: shape (B, T, C, H, W)
    state: dict with 'prev_frame' (B, C, 1, H, W) for streaming block-wise operation.
           None for batch mode (backward compatible).
    Returns:
        prune_mask, new_state  (when state is not None)
        prune_mask             (when state is None, backward compatible)
    '''


    B, T, C, H, W= x.shape
    temporal_dim, patch_size = patch_dims[0], patch_dims[1:]

    assert len(x.shape) == 5
    assert B == 1, "Batch size must be 1 for rlt_prune and unprune."
    assert T % 1 == 0, "The number of x must be even for tubelet encoding."
    assert T % temporal_dim == 0, "The number of x must be divisible by temporal_dim."
    assert H % patch_size[0] == 0, "Height must be divisible by patch_size[0]"
    assert W % patch_size[1] == 0, "Width must be divisible by patch_size[1]"

    x = x.type(torch.float32)
    x = x.transpose(1, 2)  # (B, C, T, H, W) so we can process T H W together

    # Prepend previous block's last frame for boundary diff (streaming mode)
    has_prev = state is not None and state.get('prev_frame') is not None
    if has_prev:
        x_full = torch.cat([state['prev_frame'], x], dim=2)  # (B, C, T+1, H, W)
    else:
        x_full = x

    # Save last frame for next block
    new_state = {'prev_frame': x[:, :, -1:].clone()}

    # Compare "front" of first token to "back" of second token
    # change diffs shape to B=1, C=1, T, H, W
    raw_diffs = cal_diffs(x_full[:, :, (2*temporal_dim-1)::temporal_dim], x_full[:, :, :-temporal_dim:temporal_dim], distance, patch_size=patch_size[0])
    diffs = median_blur(raw_diffs)
    if distance == "L1_gaussian":
        diffs = gaussian_blur_3D(diffs, kernel_size=(3, 3, 3), sigma=((1.5,1.5,1.5)), is_causal=True)

    # Thresholding
    prune_mask = (diffs > drop_param)
    # Skip bound_diff in streaming mode
    if drop_param2 is not None and state is None:
        prune_mask = bound_diff(prune_mask, x_full, distance, drop_param2, patch_size=2)

    prune_mask = median_blur(prune_mask.float()).bool()
    prune_mask = morphology_operation(prune_mask)
    prune_mask = dilation(prune_mask)
    if min_true_ratio is not None:
        T_mask = prune_mask.shape[2]
        tokens_per_frame = prune_mask[:, :, 0].numel()
        before_pcts = [prune_mask[:, :, t].sum().item() / tokens_per_frame * 100 for t in range(T_mask)][:15]
        print(f"[enforce_min_true_ratio] Before: {[f'{p:.1f}%' for p in before_pcts]}")
        prune_mask = enforce_min_true_ratio(prune_mask, min_true_ratio)
        after_pcts = [prune_mask[:, :, t].sum().item() / tokens_per_frame * 100 for t in range(T_mask)][:15]
        print(f"[enforce_min_true_ratio] After:  {[f'{p:.1f}%' for p in after_pcts]}")

    # append first mask frame and reshaping
    first_frame = torch.full_like(prune_mask[:, :, 0:1], bool(True))
    prune_mask = torch.cat([first_frame, prune_mask], dim=2)

    prune_mask = prune_mask.expand(-1, C, -1, -1, -1) \
                            .repeat_interleave(patch_size[0], dim=3) \
                            .repeat_interleave(patch_size[1], dim=4) \
                            .transpose(1, 2)

    # Trim prepended frame's contribution (streaming mode)
    if has_prev:
        prune_mask = prune_mask[:, 1:]

    if state is not None:
        return prune_mask, new_state
    return prune_mask

def unprune(kept_tokens, prune_mask, tubelet_shape, prev_cache=None, check_merge_place=True, use_prev_cache_only=True) -> torch.Tensor:
    """
    Recover the pruned kept_tokens by substituting the kept token with the pruned token.
    
    Args:
        kept_tokens: Tensor of kept tokens, shape (B, T, C, H, W)
        prune_mask: Bool tensor, shape (B, T, C, H, W), True for kept, False for pruned
        tubelets_seq_shape: dim of tubelets of sequance of t, h, w.
        
    Returns:
        unpruned: Tensor of shape (B=1, T, C, H, W), pruned positions filled with kept token.
  
    Steps:
    1. Expand unpruned to have the shape of (all tokens, C, p0 (temporal), p1 (h), p2 (w)).
    2. Fill in the sel_token of unpruned with kept_tokens.
    3. Fill in the missing tokens with the unpruned's (temporally) previous tubelet.
    4. reshape the unpruned to the original shape (B=1, (t h w), C, p0, p1, p2) -> (B=1, (t p0), C, (h p1), (w p2)).
    
    Note: (1) the first frame is always kept (2) batch is 1 (3) only T may change, C H W is fixed .

  """
    dtype = kept_tokens.dtype
    device = kept_tokens.device

    _, p0, p1, p2 = kept_tokens.shape[1:]
    (p0, p1, p2) = tubelet_shape

    kept_tokens = rearrange(
        kept_tokens, 
        "b (t p0) c (h p1) (w p2) -> b (t h w) c p0 p1 p2",
        p0=p0, p1=p1, p2=p2
    )

    prune_mask = rearrange(
        prune_mask, 
        "b (t p0) c (h p1) (w p2) -> b t h w c p0 p1 p2",
        p0=p0, p1=p1, p2=p2
    )
    assert prune_mask.shape[0] == 1, "Batch size must be 1 for unprune and unprune."
    prune_mask = prune_mask[0]

    # 1. Expand unpruned to have the shape of (t, h, w, C, p0 (temporal), p1 (h), p2 (w)).
    unpruned = torch.full(prune_mask.shape, -1.0, dtype=dtype, device=device)
    # 2. Fill in the sel_token of unpruned with kept_tokens.
    unpruned[prune_mask] = kept_tokens[~kept_tokens.isnan()]
    # 3. Fill in the un sel_token with the unpruned's (temporally) previous tubelet.
    if use_prev_cache_only:
        # Use prev_cache for all frames; if None, use the first frame (unpruned[0])
        fill = prev_cache if prev_cache is not None else unpruned[0]
        for i in range(unpruned.shape[0]):
            unpruned[i] = torch.where(prune_mask[i], unpruned[i], fill if not check_merge_place else 0)
    else:
        for i in range(unpruned.shape[0]):
            if check_merge_place:
                if i == 0:
                    if prev_cache is None:
                        continue
                    else:
                        unpruned[i] = torch.where(prune_mask[i], unpruned[i], 0)
                else:
                    unpruned[i] = torch.where(prune_mask[i], unpruned[i], 0)

            else:
                if i == 0:
                    if prev_cache is None:
                        continue
                    else:
                        unpruned[i] = torch.where(prune_mask[i], unpruned[i], prev_cache)
                else:
                    unpruned[i] = torch.where(prune_mask[i], unpruned[i], unpruned[i-1])
            
    #4. reshape the unpruned to the original shape (t, h ,w , C, p0, p1, p2) -> (B=1, (t p0), C, (h p1), (w p2)).
    prev_cache = unpruned[-1].clone().detach()
    
    unpruned = rearrange(
        unpruned,
        "t h w C p0 p1 p2 -> (t p0) C (h p1) (w p2)"
    ).unsqueeze(0)

    return unpruned, prev_cache

def unprune_kv(k, prev_k, prune_mask, from_memory=False) -> torch.Tensor:
    """
    Recover the pruned kept_tokens by substituting the kept token with the pruned token.
    
    Args:
        k: keys, shape (B, Token num, Head num, token dim)
        prev_k: previous keys, shape (B, Token num=1560, Head num, token dim)
        prune_mask: (B, Token num=4680)
        
    Returns:
        unprune_k: Tensor of shape (B, Token num=4680, Head num, token dim)
  
  """
    assert prune_mask.shape[0] == 1, "Batch size must be 1 for unprune and unprune."

    dtype, device = k.dtype, k.device

    unpruned_k = torch.full((1, 4680, 12, 128), -1.0, dtype=dtype, device=device) # 1, 4680, 12, 128
    unpruned_k[prune_mask] = k[0]
    unpruned_k = unpruned_k[0].reshape(3, -1, 12, 128) # 3, 1560, 12, 128
    prune_mask = prune_mask[0].reshape(3, -1)  # 3, 1560

    for i in range(unpruned_k.shape[0]):
        if i == 0:
            if prev_k is None:
                continue
            else:
                unpruned_k[i] = torch.where(prune_mask[i].view(-1, 1, 1), unpruned_k[i], prev_k[0])
        else:
            if from_memory:
                unpruned_k[i] = torch.where(prune_mask[i].view(-1, 1, 1), unpruned_k[i], prev_k[0])
            else:
                unpruned_k[i] = torch.where(prune_mask[i].view(-1, 1, 1), unpruned_k[i], unpruned_k[i-1])

    unpruned_k = unpruned_k.flatten(0, 1).unsqueeze(0)

    return unpruned_k
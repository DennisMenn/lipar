<h1 align="center">Training-Free Latent Inter-frame Pruning with Attention Recovery (LIPAR)</h1>

<p align="center">
  <a href="https://sites.google.com/utexas.edu/dennismenn/about">Dennis Menn</a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=p4pjbpsAAAAJ&hl=en">Yuedong Yang</a><sup>1</sup> ·
  <a href="https://bokun-wang.github.io/">Bokun Wang</a><sup>1</sup> ·
  <a href="https://xiwenwei.github.io/">Xiwen Wei</a><sup>1</sup> ·
  <a href="https://www.linkedin.com/in/mustafa-munir-02408/">Mustafa Munir</a><sup>1</sup> ·
  <a href="https://jeff-liangf.github.io/">Feng Liang</a><sup>2</sup> ·
  <a href="https://www.ece.utexas.edu/people/faculty/radu-marculescu">Radu Marculescu</a><sup>1</sup> ·
  <a href="https://www.chenfengx.com/">Chenfeng Xu</a><sup>1</sup> ·
  <a href="https://www.ece.utexas.edu/people/faculty/diana-marculescu">Diana Marculescu</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>The University of Texas at Austin &nbsp;&nbsp; <sup>2</sup>Meta GenAI
</p>

<p align="center">
  <a href="https://drive.google.com/file/d/1a5Oap0lnTgKfnNPWSXIKf8fi7bCgCzSx/view?usp=sharing">Paper</a> |
  Webpage (coming soon)
</p>

---

## Overview
Latent Inter-frame Pruning with Attention Recovery (LIPAR) is a **training-free** acceleration framework for diffusion transformers. It exploits temporal redundancy in latent features to avoid re-editing repeated patches while preserving visual quality through an **Attention Recovery** mechanism.

LIPAR consists of three parts:
1. **Latent Inter-frame Pruning**: skip re-computing redundant latent patches.
2. **Attention Recovery**: reduce train-inference mismatch caused by pruning.
3. **Restoration**: copy to recover full latent dimensions after denoising.

Empirically, LIPAR improves video editing throughput by **1.45×** on average (from **8.4 FPS** to **12.2 FPS** on A6000) on selected videos while preserving visual quality.

---

## Requirements

Tested setup:
- NVIDIA GPU with at least ~19GB memory (RTX 4090, A6000 tested)
- Linux
- 64GB RAM

Other hardware may work but is not officially tested.

## Installation

Install CUDA 12.8:
1. Download CUDA 12.8 from NVIDIA: https://developer.nvidia.com/cuda-12-8-0-download-archive
2. Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Create environment and install dependencies:

```bash
conda create -n lipar python=3.10 -y
conda activate lipar
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start

### 1) Download checkpoints

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

Install TAEHV-VAE: https://github.com/madebyollin/taehv/

### 2) Preprocess source videos

Before the video editing stage, preprocess the source videos:

```bash
python3 data/resize_video.py <source_video_dir> <output_dir>
```

### 3) Run LIPAR inference

```bash
python3 inference.py --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path data/davis_prompt.json \
  --output_folder outputs/lipar \
  --use_lipar
```

Original Self-Forcing inference with TAE:
```bash
python3 inference.py --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path data/davis_prompt.json \
  --output_folder outputs/no_prune \
```

Notes:
- Long, detailed prompts generally perform better.

---

## Important Files for LIPAR

- `inference.py`: main inference entry point
- `src/wan/modules/causal_model.py`: causal diffusion transformer with Attention Recovery
- `src/utils/rlt_prune.py`: Latent Inter-frame pruning and token restoration utilities
- `src/pipeline/causal_inference.py`: denoising loop and cache orchestration
- `src/utils/wan_wrapper.py`: per-step model wrapper logic

---

## Acknowledgements

This codebase is built on top of:
- [CausVid](https://github.com/tianweiy/CausVid)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)

## Citation

If you find this repository useful, please cite the paper. 
<!-- ```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
``` -->
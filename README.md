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
  <a href="https://arxiv.org/abs/2603.05811">Arxiv</a> | <a href="https://dennismenn.github.io/lipar/">Webpage</a>
</p>

---

## Overview

<table align="center" width="600">
  <tr>
    <td colspan="2" align="center">
      <a href="docs/static/images/comparisons/edited.mp4">
        <img src="docs/static/images/comparisons/edited.gif" width="600" alt="Edited comparison video preview" />
      </a>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>Self-forcing (GPU: 20.7 GB)</strong></td>
    <td width="50%" align="center"><strong>LIPAR (GPU: 16.6 GB)</strong></td>
  </tr>
</table>
<p align="center">throughput/memory evaluated on RTX 4090</p>

Latent Inter-frame Pruning with Attention Recovery (LIPAR) is a **training-free** acceleration framework for conditioned video generation using diffusion transformers. It exploits temporal redundancy in latent features to avoid re-editing unchanged patches while preserving visual quality through an **Attention Recovery** mechanism.

LIPAR consists of three parts:
1. **Latent Inter-frame Pruning**: Prune unchanged latent patches.
2. **Attention Recovery**: reduce train-inference mismatch caused by pruning.
3. **Restoration**: For decoding, recover full latent dimensions after denoising.

Empirically, LIPAR improves average video editing throughput by **1.53×** (from **12.6 FPS** to **19.3 FPS** on RTX 4090), reduces GPU memory usage, while preserving visual quality on selected DAVIS 2017 videos.

## Requirements
- NVIDIA GPU with at least 18GB memory (RTX 4090, A6000 tested)
- Linux
- 32GB RAM

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
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 2) Preprocess source videos

Before the video editing stage, resize the source videos:

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

Notes:
- For original Self-Forcing inference, remove the `--use_lipar` flag.
- Long, detailed prompts generally perform better.
- TAE weights are from https://github.com/madebyollin/taehv

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

```
@article{{menn2026trainingfreelatentinterframepruning,
  title={Training-free Latent Inter-Frame Pruning with Attention Recovery},
  author={Dennis Menn and Yuedong Yang and Bokun Wang and Xiwen Wei and Mustafa Munir and Feng Liang and Radu Marculescu and Chenfeng Xu and Diana Marculescu},
  journal={arXiv preprint arXiv:2603.05811},
  year={2026}
}
```
# StreamDiffusion for Mac

Real-time camera image-to-image transformation using diffusion models on Apple Silicon, accelerated with CoreML.

**22.7 FPS** at 512x512 resolution on Apple M3 Ultra with SDXS-512.

## Requirements

- **macOS 14+** (Sonoma or later)
- **Apple Silicon** (M1 / M2 / M3 / M4 series)
- **Python 3.9+**
- Camera (built-in or USB webcam)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ochyai/streamdiffusion-mac.git
cd streamdiffusion-mac

# 2. Setup environment
chmod +x setup.sh
./setup.sh

# 3. Activate
source .venv/bin/activate

# 4. Convert models to CoreML (one-time, ~5 minutes)
python scripts/convert_models.py

# 5. Run camera
python camera.py --prompt "oil painting style, masterpiece"
```

## Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install coremltools
pip install opencv-python numpy

# Convert models
python scripts/convert_models.py

# Run
python camera.py
```

## Usage

### Basic

```bash
# Default (SDXS-512, best speed/quality balance)
python camera.py --prompt "oil painting style, masterpiece"

# Watercolor style
python camera.py --prompt "watercolor painting, soft brushstrokes"

# With built-in prompt gallery (10 styles, press n/p to switch)
python camera.py --prompts
```

### Camera Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame |
| `n` / `p` | Next / Previous prompt |
| `+` / `-` | Adjust camera blend ratio |
| `e` / `d` | Adjust EMA smoothing |

### Advanced Options

```bash
# Use SD-Turbo instead of SDXS (slower but different style)
python camera.py --model sd-turbo --prompt "anime style"

# Blend camera with AI output (30% camera)
python camera.py --blend 0.3

# Adjust temporal smoothing
python camera.py --ema 0.9 --feedback 0.4

# Lower resolution for faster inference on smaller Macs
python camera.py --render-size 384

# Select camera device
python camera.py --camera 1
```

### Model Conversion

```bash
# Convert SDXS-512 (default, recommended)
python scripts/convert_models.py

# Convert SD-Turbo
python scripts/convert_models.py --model sd-turbo

# Custom output directory
python scripts/convert_models.py --output-dir ./my_models
python camera.py --coreml-dir ./my_models
```

## Architecture

```
Camera Thread ──→ latest_frame ──→ Inference Thread ──→ latest_ai_result
     (30 FPS)         │              (CoreML pipeline)       │
                      │                                      │
                      └──────→ Display Thread ←──────────────┘
                               (blends camera + AI, 30+ FPS)
```

The pipeline decouples camera capture, AI inference, and display into three independent threads. The inference thread runs the full CoreML pipeline on every frame:

1. **Preprocess**: Center crop, resize to 512x512, normalize
2. **VAE Encode**: Image → Latent space (CoreML TAESD, ~5ms)
3. **Noise Addition**: Fixed noise for temporal coherence
4. **UNet Inference**: Single-step denoising (CoreML, ~24ms for SDXS)
5. **VAE Decode**: Latent → Image (CoreML TAESD, ~5ms)
6. **Postprocess**: Denormalize, resize, display

Temporal coherence is maintained through:
- **Fixed noise seed**: Same noise pattern every frame eliminates flickering
- **Latent feedback**: 30% of previous frame's denoised latent blended into current input
- **EMA smoothing**: Exponential moving average on display output

## Performance

Benchmarked on Apple M3 Ultra (60-core GPU, 512GB unified memory):

| Model | Parameters | UNet Latency | Camera FPS | Quality |
|-------|-----------|-------------|------------|---------|
| **SDXS-512** | **328M** | **24.4ms** | **22.7** | Good |
| SD-Turbo | 866M | 53.2ms | 13.8 | Good |
| Tiny-SD | 323M | 31.3ms | ~20 | Fair |

> Performance scales with GPU core count. Expected approximate FPS:
> - M1/M2: ~5-8 FPS
> - M1/M2 Pro: ~8-12 FPS
> - M1/M2/M3 Max: ~12-18 FPS
> - M3 Ultra: ~22 FPS
> - M4 Max: TBD

## Experiment Report

This project is the result of a systematic 10-phase optimization study on real-time diffusion model inference on Apple Silicon. Below is a summary of key findings.

### What Works on Apple Silicon

| Technique | Effect | Notes |
|-----------|--------|-------|
| **CoreML conversion** | **+64%** | Only effective UNet acceleration method |
| **Distilled models (SDXS)** | **+118%** | Best speed/quality trade-off |
| 3-thread pipeline | Smooth display | Decouples inference from rendering |

### What Does NOT Work on Apple Silicon

| Technique | Effect | Why |
|-----------|--------|-----|
| Quantization (INT8 to 2-bit) | 0% | M3 Ultra is compute-bound, not memory-bandwidth-bound |
| Token Merging (ToMe) | -10% | MPS overhead exceeds attention savings |
| Parallel CoreML inference | 0% | Metal serializes GPU commands |
| Neural Engine for UNet | -19% to -520% | ANE unsuitable for large (866M) models |
| torch.compile | Crash | MPS backend not supported |
| Attention Slicing | -40% | MPS memory management overhead |

### Key Insight: CUDA Optimization Wisdom Does Not Apply

The most important finding is that optimization techniques established for NVIDIA GPUs and the CUDA ecosystem largely **do not transfer** to Apple Silicon's unified memory architecture:

- **Quantization is ineffective** because Apple Silicon is compute-bound (not memory-bandwidth-bound). The 800 GB/s unified memory bandwidth is sufficient for model weights, so reducing precision doesn't help.
- **Parallel inference is impossible** because CoreML serializes Metal GPU commands, unlike CUDA Streams which allow fine-grained kernel-level parallelism.
- **The software ecosystem is immature** compared to CUDA's decades of optimization (cuDNN, TensorRT, xformers, Flash Attention). torch.compile doesn't work on MPS, and many PyTorch operations have suboptimal Metal implementations.

### Negative Results

Several creative approaches were also tested and yielded negative results:

- **kNN search-based synthesis** (Phase 7): 512GB memory enables searching 100M vectors in 0.5ms, but kNN retrieval fundamentally cannot replace the continuous nonlinear function approximation of a UNet.
- **pix2pix-turbo** (Phase 8): Skip-connection VAE design prevents CoreML conversion, creating a 160ms VAE bottleneck (vs 53ms UNet). Result: 4 FPS.
- **Optical flow frame skipping** (Phase 9): Warping between UNet frames produces jelly-like distortion artifacts. 17.4 FPS, worse than SDXS baseline.
- **Knowledge distillation** (Phase 10): 875K-parameter feedforward CNN trained with L1 loss produces blank output. The capacity gap vs 328M-parameter diffusion model is too large.

### Full Paper

A detailed academic paper covering all experiments is available:
- [paper.tex](paper.tex) (Japanese)
- [paper_en.tex](paper_en.tex) (English)

## License

MIT License

## Citation

```bibtex
@article{ochiai2025streamdiffusion_mac,
  title={Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra},
  author={Ochiai, Yoichi},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) — Original pipeline architecture
- [SDXS](https://github.com/IDKiro/sdxs) — Distillation-specialized model
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) — One-step diffusion baseline
- [TAESD](https://github.com/madebyollin/taesd) — Tiny Autoencoder for Stable Diffusion

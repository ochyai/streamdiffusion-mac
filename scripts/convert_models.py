#!/usr/bin/env python3
"""
CoreML Model Conversion Script for StreamDiffusion-Mac

Converts SDXS-512 UNet + TinyVAE (encoder/decoder) to CoreML format.
SDXS-512 is the optimal model for real-time inference on Apple Silicon,
achieving 22.7 FPS camera img2img on M3 Ultra.

Usage:
    python scripts/convert_models.py
    python scripts/convert_models.py --output-dir ./coreml_models
    python scripts/convert_models.py --model sd-turbo   # Use SD-Turbo instead
"""
import os
import sys
import time
import gc
import argparse
import numpy as np
import torch
import coremltools as ct

MODEL_CONFIGS = {
    "sdxs": {
        "model_id": "IDKiro/sdxs-512-0.9",
        "hidden_size": 1024,
        "unet_name": "unet_sdxs_512",
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "hidden_size": 1024,
        "unet_name": "unet_sd_turbo",
    },
}


def convert_unet(model_id, hidden_size, save_path):
    """Convert UNet to CoreML."""
    from diffusers import StableDiffusionPipeline

    print(f"  Loading UNet from {model_id}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
    except ValueError:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    unet = pipe.unet.eval().float().cpu()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample, timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

    wrapper = UNetWrapper(unet).eval()

    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([999.0])
    hidden_states = torch.randn(1, 77, hidden_size)

    print("  Tracing UNet...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, timestep, hidden_states))

    print("  Converting to CoreML (this may take several minutes)...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape, dtype=np.float16),
            ct.TensorType(name="timestep", shape=timestep.shape, dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="noise_pred", dtype=np.float16),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  UNet converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    # Return pipe for text encoder conversion
    del unet, wrapper, traced, model
    gc.collect()
    return pipe


def convert_vae_encoder(save_path, size=512):
    """Convert TinyVAE Encoder to CoreML."""
    from diffusers import AutoencoderTiny

    print(f"  Loading TinyVAE...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder

        def forward(self, x):
            return self.encoder(x)

    wrapper = Wrapper(vae).eval()
    dummy = torch.randn(1, 3, size, size)

    print("  Tracing VAE Encoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("  Converting to CoreML...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  VAE Encoder converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    del vae, wrapper, traced, model
    gc.collect()


def convert_vae_decoder(save_path, size=512):
    """Convert TinyVAE Decoder to CoreML."""
    from diffusers import AutoencoderTiny

    print(f"  Loading TinyVAE...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder

        def forward(self, x):
            return self.decoder(x)

    wrapper = Wrapper(vae).eval()
    latent_size = size // 8
    dummy = torch.randn(1, 4, latent_size, latent_size)

    print("  Tracing VAE Decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("  Converting to CoreML...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  VAE Decoder converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    del vae, wrapper, traced, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Convert models to CoreML")
    parser.add_argument("--output-dir", default="coreml_models",
                        help="Output directory for CoreML models")
    parser.add_argument("--model", default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to convert (default: sdxs)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg = MODEL_CONFIGS[args.model]

    print("=" * 60)
    print(f"CoreML Model Conversion — {args.model}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    # Step 1: UNet
    print(f"\n[1/3] Converting UNet ({cfg['model_id']})...")
    unet_path = os.path.join(output_dir, f"{cfg['unet_name']}.mlpackage")
    pipe = convert_unet(cfg["model_id"], cfg["hidden_size"], unet_path)

    # Step 2: VAE Encoder
    print(f"\n[2/3] Converting TinyVAE Encoder (512x512)...")
    enc_path = os.path.join(output_dir, "taesd_encoder_512.mlpackage")
    convert_vae_encoder(enc_path, size=512)

    # Step 3: VAE Decoder
    print(f"\n[3/3] Converting TinyVAE Decoder (512x512)...")
    dec_path = os.path.join(output_dir, "taesd_decoder.mlpackage")
    convert_vae_decoder(dec_path, size=512)

    print("\n" + "=" * 60)
    print("ALL CONVERSIONS COMPLETE!")
    print(f"Models saved to: {os.path.abspath(output_dir)}")
    print("")
    print("Next: Run the camera pipeline:")
    print(f"  python camera.py --prompt 'oil painting style'")
    print("=" * 60)


if __name__ == "__main__":
    main()

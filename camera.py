#!/usr/bin/env python3
"""
StreamDiffusion for Mac — Real-time Camera img2img Pipeline

CoreML-accelerated real-time image-to-image transformation using
diffusion models on Apple Silicon. Achieves 22.7 FPS on M3 Ultra
with SDXS-512 at 512x512 resolution.

Architecture:
  - Camera thread:    captures frames continuously
  - Inference thread: runs full CoreML pipeline (VAE enc → UNet → VAE dec)
  - Display thread:   blends camera + AI output at 30+ FPS

Usage:
    python camera.py --prompt "oil painting style, masterpiece"
    python camera.py --prompt "watercolor painting" --blend 0.3
    python camera.py --model sd-turbo --prompt "anime style"

Controls:
    q     : quit
    s     : save current frame
    n / p : next / previous prompt
    + / - : adjust camera blend ratio
    e / d : adjust EMA smoothing
"""
import os
import sys
import time
import gc
import argparse
import threading
import numpy as np
import cv2
import coremltools as ct

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_models")

DEFAULT_PROMPTS = [
    "oil painting style, masterpiece, highly detailed",
    "watercolor painting, soft brushstrokes, artistic, beautiful colors",
    "pencil sketch, detailed line art, cross-hatching, artistic",
    "cyberpunk neon city, futuristic, glowing lights, sci-fi",
    "studio ghibli anime style, beautiful landscape, fantasy",
    "photorealistic HDR photograph, dramatic lighting, sharp detail",
    "impressionist painting, Monet style, soft light, garden scene",
    "dark fantasy art, dramatic chiaroscuro, gothic atmosphere",
    "ukiyo-e Japanese woodblock print style, bold lines, flat colors",
    "pop art style, bold colors, comic book aesthetic, Roy Lichtenstein",
]

MODEL_CONFIGS = {
    "sdxs": {
        "model_id": "IDKiro/sdxs-512-0.9",
        "hidden_size": 1024,
        "unet_prefix": "unet_sdxs_512",
        "scheduler": "euler",
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "hidden_size": 1024,
        "unet_prefix": "unet_sd_turbo",
        "scheduler": "default",
    },
}


def ensure_vae_encoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Encoder if not present."""
    path = os.path.join(coreml_dir, f"taesd_encoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    print(f"  Auto-converting TinyVAE Encoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder
        def forward(self, x):
            return self.encoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 3, render_size, render_size)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    print(f"  Saved: {path}")
    return path


def ensure_vae_decoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Decoder if not present."""
    if render_size == 512:
        path = os.path.join(coreml_dir, "taesd_decoder.mlpackage")
    else:
        path = os.path.join(coreml_dir, f"taesd_decoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    ls = render_size // 8
    print(f"  Auto-converting TinyVAE Decoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder
        def forward(self, x):
            return self.decoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 4, ls, ls)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    print(f"  Saved: {path}")
    return path


class Pipeline:
    """CoreML img2img pipeline with temporal coherence."""

    def __init__(self, model_name, render_size, output_size, prompt,
                 strength=0.5, prompts=None, latent_feedback=0.3, coreml_dir=COREML_DIR):
        import torch
        self.model_name = model_name
        self.render_size = render_size
        self.output_size = output_size
        self.latent_size = render_size // 8
        cfg = MODEL_CONFIGS[model_name]

        print("\n--- Loading CoreML Models ---")
        enc_path = ensure_vae_encoder(render_size, coreml_dir)
        dec_path = ensure_vae_decoder(render_size, coreml_dir)
        prefix = cfg["unet_prefix"]
        unet_path = os.path.join(coreml_dir, f"{prefix}.mlpackage")
        if not os.path.exists(unet_path):
            print(f"ERROR: UNet model not found at {unet_path}")
            print(f"  Run: python scripts/convert_models.py --model {model_name}")
            sys.exit(1)
        print(f"  UNet: {unet_path}")

        cu = ct.ComputeUnit.CPU_AND_GPU
        self.vae_encoder = ct.models.MLModel(enc_path, compute_units=cu)
        self.vae_decoder = ct.models.MLModel(dec_path, compute_units=cu)
        self.unet = ct.models.MLModel(unet_path, compute_units=cu)

        # Buffers
        self._img_buf = np.empty((1, 3, render_size, render_size), dtype=np.float16)
        self._lat_buf = np.empty((1, 4, self.latent_size, self.latent_size), dtype=np.float16)
        self._out_buf = np.empty((1, 4, self.latent_size, self.latent_size), dtype=np.float16)
        self._t_buf = np.empty((1,), dtype=np.float16)
        self._norm_lut = (np.arange(256, dtype=np.float32) / 127.5 - 1.0).astype(np.float16)

        # Prompt encoding
        print("  Loading text encoder...")
        pipe = __import__('diffusers').StableDiffusionPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16).to("mps")
        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder

        # Scheduler setup
        if cfg["scheduler"] == "euler":
            from diffusers import EulerDiscreteScheduler
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(1 if cfg["scheduler"] == "euler" else 50, device="mps")

        if cfg["scheduler"] == "euler":
            actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[
                min(int(actual_t), len(pipe.scheduler.alphas_cumprod) - 1)
            ].item()
        else:
            t_idx = max(0, int(50 * (1.0 - strength)))
            if t_idx < len(pipe.scheduler.timesteps):
                actual_t = pipe.scheduler.timesteps[t_idx].cpu().item()
            else:
                actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[int(actual_t)].item()

        self._sqrt_a = np.float16(np.sqrt(ap))
        self._sqrt_1ma = np.float16(np.sqrt(1.0 - ap))

        # Encode all prompts
        self._all_prompts = prompts if prompts else [prompt]
        self._prompt_index = 0
        self._all_embeds = []
        print(f"  Encoding {len(self._all_prompts)} prompt(s)...")
        for p in self._all_prompts:
            self._all_embeds.append(self._encode_single(p))
        self._prompt_embeds = self._all_embeds[0]
        self._current_prompt = self._all_prompts[0]
        self._target_embeds = self._prompt_embeds.copy()
        self._prompt_lerp_speed = 0.05

        del pipe
        gc.collect()
        torch.mps.empty_cache()

        # Fixed noise for temporal coherence
        rng = np.random.RandomState(42)
        self._fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(np.float16)
        self._prev_denoised = None
        self.latent_feedback = latent_feedback

        print("  Warming up CoreML...")
        self._warmup()
        print("  Ready!")

    def _encode_single(self, prompt):
        import torch
        with torch.no_grad():
            ti = self._tokenizer(
                prompt, padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            return self._text_encoder(ti.input_ids.to("mps"))[0].cpu().to(torch.float16).numpy()

    def next_prompt(self):
        self._prompt_index = (self._prompt_index + 1) % len(self._all_prompts)
        self._target_embeds = self._all_embeds[self._prompt_index]
        self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def prev_prompt(self):
        self._prompt_index = (self._prompt_index - 1) % len(self._all_prompts)
        self._target_embeds = self._all_embeds[self._prompt_index]
        self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def _warmup(self, n=25):
        np.copyto(self._img_buf, np.random.randn(1, 3, self.render_size, self.render_size).astype(np.float16))
        for _ in range(n):
            e = self.vae_encoder.predict({"image": self._img_buf})
            np.copyto(self._lat_buf, np.array(e["latent"]).astype(np.float16))
            u = self.unet.predict({
                "sample": self._lat_buf, "timestep": self._t_buf,
                "encoder_hidden_states": self._prompt_embeds,
            })
            np.copyto(self._out_buf, np.array(u["noise_pred"]).astype(np.float16))
            self.vae_decoder.predict({"latent": self._out_buf})

    def process_frame(self, frame_bgr):
        """Full pipeline: preprocess → VAE enc → UNet → VAE dec → postprocess."""
        h, w = frame_bgr.shape[:2]
        if w > h:
            off = (w - h) // 2
            frame_bgr = frame_bgr[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            frame_bgr = frame_bgr[off:off + w, :]

        resized = cv2.resize(frame_bgr, (self.render_size, self.render_size))
        rgb = resized[:, :, ::-1]
        np.copyto(self._img_buf, self._norm_lut[rgb].transpose(2, 0, 1)[np.newaxis])

        # Smooth prompt transition
        diff = self._target_embeds - self._prompt_embeds
        if np.abs(diff).max() > 1e-4:
            self._prompt_embeds = self._prompt_embeds + self._prompt_lerp_speed * diff

        # VAE Encode
        enc = self.vae_encoder.predict({"image": self._img_buf})
        clean = np.array(enc["latent"]).astype(np.float16)

        # Latent feedback from previous frame
        if self._prev_denoised is not None and self.latent_feedback > 0:
            fb = np.float16(self.latent_feedback)
            clean = (1.0 - fb) * clean + fb * self._prev_denoised

        # Add fixed noise (same every frame for temporal coherence)
        noisy = self._sqrt_a * clean + self._sqrt_1ma * self._fixed_noise
        np.copyto(self._lat_buf, noisy)

        # UNet inference
        u = self.unet.predict({
            "sample": self._lat_buf, "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        })
        npred = np.array(u["noise_pred"]).astype(np.float16)
        denoised = (noisy - self._sqrt_1ma * npred) / self._sqrt_a

        self._prev_denoised = denoised.copy()
        np.copyto(self._out_buf, denoised)

        # VAE Decode
        dec = self.vae_decoder.predict({"latent": self._out_buf})
        r = np.array(dec["image"]).astype(np.float32).squeeze(0).transpose(1, 2, 0)
        r = ((r + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if r.shape[0] != self.output_size:
            r = cv2.resize(r, (self.output_size, self.output_size))
        return cv2.cvtColor(r, cv2.COLOR_RGB2BGR)


class CameraApp:
    """3-thread camera application for smooth real-time display."""

    def __init__(self, pipeline, camera_id=0, blend_ratio=0.0, ema_alpha=0.85):
        self.pipeline = pipeline
        self.camera_id = camera_id
        self.blend_ratio = blend_ratio
        self.ema_alpha = ema_alpha
        self.running = False

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._result_lock = threading.Lock()
        self._latest_ai_result = None
        self._ai_update_count = 0
        self._ai_fps = 0.0
        self._ema_result = None

    def _camera_thread(self, cap):
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame

    def _inference_thread(self):
        fps_times = []
        while self.running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            result = self.pipeline.process_frame(frame)
            elapsed = time.perf_counter() - t0

            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)

            with self._result_lock:
                self._latest_ai_result = result
                self._ai_update_count += 1
                self._ai_fps = 1.0 / (sum(fps_times) / len(fps_times))

    def run(self):
        print(f"\nOpening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)
        else:
            print("ERROR: Cannot read frames!")
            cap.release()
            return

        out_sz = self.pipeline.output_size
        print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Blend: {self.blend_ratio:.1f} (0=AI only, 1=camera only)")
        print(f"EMA: {self.ema_alpha:.2f}")
        print("Controls: q=quit  s=save  n/p=prompt  +/-=blend  e/d=EMA")
        print("")

        self.running = True
        cam_t = threading.Thread(target=self._camera_thread, args=(cap,), daemon=True)
        inf_t = threading.Thread(target=self._inference_thread, daemon=True)
        cam_t.start()
        inf_t.start()

        win = f"StreamDiffusion-Mac ({self.pipeline.model_name} {self.pipeline.render_size})"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, out_sz * 2 + 20, out_sz)

        display_count = 0
        total_start = time.perf_counter()
        display_fps_times = []

        while self.running:
            dt0 = time.perf_counter()

            with self._frame_lock:
                frame = self._latest_frame
            with self._result_lock:
                ai_result = self._latest_ai_result
                ai_fps = self._ai_fps

            if frame is None:
                time.sleep(0.001)
                continue

            # Camera display
            h, w = frame.shape[:2]
            if w > h:
                off = (w - h) // 2
                fsq = frame[:, off:off + h]
            elif h > w:
                off = (h - w) // 2
                fsq = frame[off:off + w, :]
            else:
                fsq = frame
            cam_display = cv2.resize(fsq, (out_sz, out_sz))

            if ai_result is not None:
                if self._ema_result is None:
                    self._ema_result = ai_result.astype(np.float32)
                else:
                    self._ema_result = (
                        self.ema_alpha * self._ema_result
                        + (1.0 - self.ema_alpha) * ai_result.astype(np.float32)
                    )
                smooth_ai = self._ema_result.clip(0, 255).astype(np.uint8)

                if self.blend_ratio > 0.01:
                    result_display = cv2.addWeighted(
                        smooth_ai, 1.0 - self.blend_ratio,
                        cam_display, self.blend_ratio, 0,
                    )
                else:
                    result_display = smooth_ai
            else:
                result_display = cam_display

            display = np.hstack([cam_display, result_display])

            display_count += 1
            dt = time.perf_counter() - dt0
            display_fps_times.append(dt)
            if len(display_fps_times) > 60:
                display_fps_times.pop(0)

            pidx = self.pipeline._prompt_index + 1
            ptotal = len(self.pipeline._all_prompts)
            ptext = self.pipeline._current_prompt[:70]

            cv2.putText(display, f"AI: {ai_fps:.1f} FPS | Prompt {pidx}/{ptotal}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, ptext,
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
            cv2.putText(display, "Camera",
                        (out_sz // 2 - 40, out_sz - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "AI Output",
                        (out_sz + out_sz // 2 - 55, out_sz - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                fn = f"capture_{int(time.time())}.png"
                cv2.imwrite(fn, result_display)
                print(f"  Saved: {fn}")
            elif key == ord('n'):
                p = self.pipeline.next_prompt()
                print(f"  [{pidx}/{ptotal}] {p[:60]}")
            elif key == ord('p'):
                p = self.pipeline.prev_prompt()
                print(f"  [{pidx}/{ptotal}] {p[:60]}")
            elif key in (ord('+'), ord('=')):
                self.blend_ratio = min(1.0, self.blend_ratio + 0.05)
                print(f"  Blend: {self.blend_ratio:.2f}")
            elif key == ord('-'):
                self.blend_ratio = max(0.0, self.blend_ratio - 0.05)
                print(f"  Blend: {self.blend_ratio:.2f}")
            elif key == ord('e'):
                self.ema_alpha = min(0.99, self.ema_alpha + 0.05)
                print(f"  EMA: {self.ema_alpha:.2f}")
            elif key == ord('d'):
                self.ema_alpha = max(0.0, self.ema_alpha - 0.05)
                print(f"  EMA: {self.ema_alpha:.2f}")

        self.running = False
        cam_t.join(timeout=2)
        inf_t.join(timeout=2)

        total = time.perf_counter() - total_start
        with self._result_lock:
            ai_total = self._ai_update_count
        print(f"\nSession: {display_count} display frames in {total:.1f}s = {display_count / total:.1f} display FPS")
        print(f"  AI inference: {ai_total} frames = {ai_total / total:.1f} AI FPS")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — Real-time Camera")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true",
                        help="Use built-in prompt gallery (10 styles)")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: sdxs for best performance)")
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512])
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Camera blend (0.0=AI only, 0.3=30%% camera)")
    parser.add_argument("--ema", type=float, default=0.85,
                        help="EMA smoothing (0=none, 0.9=heavy)")
    parser.add_argument("--feedback", type=float, default=0.3,
                        help="Latent feedback (0=none, 0.3=30%% prev frame)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print(f"StreamDiffusion for Mac — {args.model} {args.render_size}x{args.render_size}")
    print("  CoreML-accelerated real-time camera img2img")
    print("=" * 60)

    prompts = DEFAULT_PROMPTS if args.prompts else None

    pipeline = Pipeline(
        model_name=args.model,
        render_size=args.render_size,
        output_size=args.output_size,
        prompt=args.prompt,
        strength=args.strength,
        prompts=prompts,
        latent_feedback=args.feedback,
        coreml_dir=args.coreml_dir,
    )

    app = CameraApp(
        pipeline=pipeline,
        camera_id=args.camera,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
    )
    app.run()


if __name__ == "__main__":
    main()

"""SeedVR2 refinement of Lada's restored mosaic crops.

Lada cuts every mosaic region into a fixed-size (256x256) crop clip, restores it
with BasicVSR++ and then pastes it back into the full frame. The hook below sits
between those two steps: the restored crop clip is upscaled/refined by SeedVR2
(one-step diffusion video restoration), and the clip's padding and masks are
rescaled so Lada's own blending code can paste the sharper crop back. Nothing
outside the mosaic crops is ever sent through SeedVR2.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass

import cv2
import torch

from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline.mosaic_detector import Clip
from lada.utils import image_utils

logger = logging.getLogger(__name__)


@dataclass
class SeedVR2Options:
    repo_dir: str
    model_dir: str
    dit_model: str = "seedvr2_ema_3b_fp16.safetensors"
    batch_size: int = 21           # frames per diffusion batch; must be 4n+1
    color_correction: str = "lab"
    seed: int = 42
    blocks_to_swap: int = 0
    attention_mode: str = "sdpa"
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    vae_tile_size: int = 0         # >0: tiled VAE encode/decode with this tile size (px), saves memory
    debug: bool = False


def _load_inference_module(repo_dir: str):
    """Import SeedVR2's inference_cli.py as a library (it puts its own ``src`` on sys.path)."""
    path = os.path.join(repo_dir, "inference_cli.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SeedVR2 not found at {repo_dir}. Run ./setup.sh --with-seedvr2")
    spec = importlib.util.spec_from_file_location("seedvr2_inference_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeedVR2Refiner:
    """Keeps the SeedVR2 DiT/VAE loaded and refines short BGR uint8 clips."""

    def __init__(self, opts: SeedVR2Options):
        if opts.batch_size < 1 or (opts.batch_size - 1) % 4 != 0:
            raise ValueError(f"SeedVR2 batch size must be 4n+1 (1, 5, 9, ... 21, ...), got {opts.batch_size}")
        self.opts = opts
        self._cli = _load_inference_module(opts.repo_dir)
        self._cli.debug.enabled = opts.debug
        self._runner_cache: dict = {}
        self._lock = threading.Lock()
        self.args = self._build_args()

    def _build_args(self) -> argparse.Namespace:
        # Reuse SeedVR2's own argument parser so every option it expects has its default.
        o = self.opts
        argv = [
            "seedvr2", "unused-input",
            "--model_dir", o.model_dir,
            "--dit_model", o.dit_model,
            "--batch_size", str(o.batch_size),
            "--color_correction", o.color_correction,
            "--seed", str(o.seed),
            "--attention_mode", o.attention_mode,
            "--input_noise_scale", str(o.input_noise_scale),
            "--latent_noise_scale", str(o.latent_noise_scale),
            # keep models resident between clips
            "--cache_dit", "--cache_vae",
        ]
        if o.vae_tile_size > 0:
            overlap = str(max(16, o.vae_tile_size // 8))
            argv += ["--vae_encode_tiled", "--vae_encode_tile_size", str(o.vae_tile_size),
                     "--vae_encode_tile_overlap", overlap,
                     "--vae_decode_tiled", "--vae_decode_tile_size", str(o.vae_tile_size),
                     "--vae_decode_tile_overlap", overlap]
        if o.blocks_to_swap > 0:
            argv += ["--blocks_to_swap", str(o.blocks_to_swap), "--dit_offload_device", "cpu"]
        saved = sys.argv
        try:
            sys.argv = argv
            return self._cli.parse_arguments()
        finally:
            sys.argv = saved

    def download_models(self) -> None:
        os.makedirs(self.opts.model_dir, exist_ok=True)
        ok = self._cli.download_weight(dit_model=self.opts.dit_model, vae_model=self._cli.DEFAULT_VAE,
                                       model_dir=self.opts.model_dir, debug=self._cli.debug)
        if not ok:
            raise RuntimeError("Failed to download SeedVR2 model weights")

    def refine(self, frames: list[torch.Tensor], out_size: int) -> list[torch.Tensor]:
        """Refine a clip of square BGR uint8 (H, W, C) frames to out_size x out_size.

        Returned frames are BGR uint8 on the device of the input frames.
        """
        device = frames[0].device
        n = len(frames)
        # Lada: BGR uint8 HWC  ->  SeedVR2: RGB float32 [T, H, W, C] in [0, 1]
        x = torch.stack([f.to("cpu") for f in frames]).flip(-1).to(torch.float32).div_(255.0)

        self.args.resolution = out_size
        self.args.max_resolution = out_size
        with self._lock:
            y = self._cli._process_frames_core(frames_tensor=x, args=self.args, device_id="0",
                                               debug=self._cli.debug, runner_cache=self._runner_cache)
        if y.shape[0] < n:
            raise RuntimeError(f"SeedVR2 returned {y.shape[0]} frames for a clip of {n}")
        y = y[:n, ..., :3]
        y = y.mul(255.0).round_().clamp_(0, 255).to(torch.uint8).flip(-1)  # RGB -> BGR

        out = []
        for f in torch.unbind(y, 0):
            if f.shape[0] != out_size or f.shape[1] != out_size:
                f = torch.from_numpy(cv2.resize(f.contiguous().numpy(), (out_size, out_size),
                                                interpolation=cv2.INTER_AREA))
            out.append(f.contiguous().to(device))
        return out


class SeedVR2FrameRestorer(FrameRestorer):
    """Lada FrameRestorer that runs SeedVR2 on each restored crop clip before it is blended back."""

    def __init__(self, *args, refiner: SeedVR2Refiner, scale: str | int = "auto", max_crop_res: int = 1024,
                 min_scale: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.refiner = refiner
        self.scale = scale
        self.max_crop_res = max_crop_res
        self.min_scale = min_scale
        self.clips_refined = 0

    def _pick_scale(self, clip: Clip) -> int:
        """Integer scale so pads stay exact; auto = enough to cover the largest original crop."""
        max_scale = max(1, self.max_crop_res // clip.size)
        if self.scale != "auto":
            return max(1, min(int(self.scale), max_scale))
        largest = max(max(shape[0], shape[1]) for shape in clip.crop_shapes)
        needed = math.ceil(largest / clip.size)
        return max(1, min(max(needed, self.min_scale), max_scale))

    def _restore_clip(self, clip: Clip):
        super()._restore_clip(clip)  # Lada BasicVSR++ restoration at clip.size (256)
        if self.mosaic_detection or len(clip.frames) == 0:
            return

        s = self._pick_scale(clip)
        out_size = clip.size * s
        clip.frames = self.refiner.refine(clip.frames, out_size)
        if s != 1:
            clip.masks = [image_utils.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
                          for m in clip.masks]
            clip.pad_after_resizes = [tuple(p * s for p in pad) for pad in clip.pad_after_resizes]
        self.clips_refined += 1
        logger.info("SeedVR2 refined clip %s (%d frames) at %dx%d", clip.id, len(clip.frames), out_size, out_size)

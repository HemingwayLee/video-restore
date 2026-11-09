"""Smoke test for the SeedVR2 crop hook (needs ./setup.sh --with-seedvr2).

Builds a synthetic Lada clip (a mosaic region in a frame), runs Lada + SeedVR2 on
it via SeedVR2FrameRestorer._restore_clip, then blends it back with Lada's
_restore_frame and checks that only the mosaic region changed.

    .venv/bin/python tests/test_seedvr2_hook.py [--scale 2]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_restore.cli as cli  # noqa: E402  (sets LADA_MODEL_WEIGHTS_DIR / MPS env before torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lada import ModelFiles  # noqa: E402
from lada.restorationpipeline import load_models  # noqa: E402
from lada.restorationpipeline.mosaic_detector import Clip, Scene  # noqa: E402
from lada.utils.os_utils import get_default_torch_device  # noqa: E402

from video_restore.seedvr2_refiner import SeedVR2FrameRestorer, SeedVR2Options, SeedVR2Refiner  # noqa: E402


class _Meta:  # minimal stand-in for lada VideoMetadata
    def __init__(self, w, h):
        self.video_width, self.video_height, self.video_file = w, h, "synthetic"


def make_frames(n, h, w, box):
    t, l, b, r = box
    rng = np.random.default_rng(0)
    base = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    frames, masks = [], []
    for i in range(n):
        f = np.roll(base, i * 2, axis=1).copy()
        roi = f[t:b + 1, l:r + 1]
        small = roi[::16, ::16]
        f[t:b + 1, l:r + 1] = np.repeat(np.repeat(small, 16, 0), 16, 1)[: roi.shape[0], : roi.shape[1]]
        m = np.zeros((h, w, 1), np.uint8)
        m[t:b + 1, l:r + 1] = 255
        frames.append(torch.from_numpy(f))
        masks.append(torch.from_numpy(m))
    return frames, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="auto")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--model", default="seedvr2_ema_3b_fp16.safetensors")
    a = ap.parse_args()

    device = torch.device(get_default_torch_device())
    h, w, box = 720, 1280, (200, 400, 599, 899)  # 400x500 mosaic region
    frames, masks = make_frames(a.frames, h, w, box)
    frames = [f.to(device) for f in frames]
    masks = [m.to(device) for m in masks]

    scene = Scene("synthetic", _Meta(w, h))
    for i, (f, m) in enumerate(zip(frames, masks)):
        scene.add_frame(i, f, m, box)

    det = ModelFiles.get_detection_model_by_name("v4-fast").path
    rest = ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2").path
    det_model, rest_model, pad_mode = load_models(device, "basicvsrpp-v1.2", rest, None, det, True, False)
    clip = Clip(scene, 256, pad_mode, id=0)

    refiner = SeedVR2Refiner(SeedVR2Options(repo_dir=cli.SEEDVR2_REPO, model_dir=cli.SEEDVR2_WEIGHTS,
                                            dit_model=a.model, batch_size=5))
    refiner.download_models()

    # Bypass FrameRestorer.__init__ (needs a real video file); set only what _restore_clip/_restore_frame use.
    r = SeedVR2FrameRestorer.__new__(SeedVR2FrameRestorer)
    r.device, r.mosaic_detection = device, False
    r.mosaic_restoration_model_name, r.mosaic_restoration_model = "basicvsrpp-v1.2", rest_model
    r.refiner, r.scale, r.max_crop_res, r.min_scale, r.clips_refined = refiner, a.scale, 1024, 2, 0

    s = r._pick_scale(clip)
    t0 = time.time()
    r._restore_clip(clip)
    dt = time.time() - t0
    assert clip.frames[0].shape == (256 * s, 256 * s, 3), clip.frames[0].shape
    assert clip.masks[0].shape[:2] == (256 * s, 256 * s), clip.masks[0].shape
    print(f"refined {a.frames} frames at {256 * s}px (scale {s}) in {dt:.1f}s")

    original = frames[0].clone()
    ct, cl, cb, cr = clip.boxes[0]  # Lada's expanded crop box: the only region it may write to
    r._restore_frame(frames[0], 0, [clip])
    diff = (frames[0].float() - original.float()).abs().cpu()
    t, l, b, rr = box
    inside = diff[t:b + 1, l:rr + 1].mean().item()
    outside_mask = torch.ones(h, w, dtype=torch.bool)
    outside_mask[ct:cb + 1, cl:cr + 1] = False
    outside = diff[outside_mask].mean().item()
    print(f"crop box {(ct, cl, cb, cr)} for mosaic box {box}")
    print(f"mean |change| inside mosaic: {inside:.2f}, outside crop box: {outside:.4f}")
    assert inside > 1.0 and outside == 0.0

    os.makedirs("test_output", exist_ok=True)
    import cv2
    cv2.imwrite("test_output/hook_before.png", original.cpu().numpy())
    cv2.imwrite("test_output/hook_after.png", frames[0].cpu().numpy())
    print("OK  (wrote test_output/hook_before.png, hook_after.png)")


if __name__ == "__main__":
    main()

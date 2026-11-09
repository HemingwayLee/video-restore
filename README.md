# video-restore

Command-line de-mosaic tool built on [Lada](https://codeberg.org/ladaapp/lada), with optional
[SeedVR2-3B](https://github.com/ByteDance-Seed/SeedVR) refinement of the restored regions.

| Mode | Command | What runs |
|------|---------|-----------|
| **Lada only** (default, fast) | `./vrestore in.mp4` | Lada detects mosaics, restores them with BasicVSR++, and blends them back. |
| **Lada + SeedVR2** | `./vrestore in.mp4 --seedvr2` | Same as above. Each restored crop clip also goes through SeedVR2-3B (one-step diffusion video restoration) before blending. **Only the mosaic crops** go through SeedVR2, never the full frame. |

## Setup

Requires macOS (Apple Silicon, MPS) or Linux (NVIDIA CUDA). `setup.sh` installs any missing
`ffmpeg`, `uv`, or `git` with Homebrew. It creates `.venv` (Python 3.12), installs pinned Lada
(and SeedVR2) revisions into `vendor/`, and downloads checksum-verified weights into `models/`.

```bash
./setup.sh                 # Lada only (~200 MB of weights)
./setup.sh --with-seedvr2  # + SeedVR2 code, deps and 3B fp16 weights (~7 GB)
```

## Usage

```bash
./vrestore movie.mp4                              # -> movie.restored.mp4
./vrestore movie.mp4 -o out.mp4 --detection-model v4-accurate
./vrestore videos/ -o restored/                   # whole directory

./vrestore movie.mp4 --seedvr2                    # -> movie.restored-seedvr2.mp4
./vrestore movie.mp4 --seedvr2 --seedvr2-scale 2 --seedvr2-batch-size 33
./vrestore --help
```

Useful options:

* `--seedvr2-scale auto|1|2|3|4`: Lada restores crops at 256 px. SeedVR2 refines them at
  256×scale px, and the result is scaled back to the original crop size. `auto` picks the
  smallest scale that covers the original crop size (minimum 2), capped by `--seedvr2-max-res`
  (default 1024).
* `--seedvr2-batch-size`: frames per diffusion batch (4n+1). Larger values give better
  temporal consistency but use more memory. Default 21.
* `--seedvr2-vae-tile 512`: tiled VAE encode/decode, which lowers peak memory for large crops
  (the VAE decode is the memory peak). On macOS, MPS allocations are capped at Metal's
  recommended working set, so running out of memory raises an error instead of exhausting
  the machine. Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` to override the cap.
* `--seedvr2-model`: a smaller 3B variant such as `seedvr2_ema_3b-Q8_0.gguf` uses less memory.
  The fp8 variant needs CUDA.
* `--encoding-preset` / `--list-encoding-presets`: Lada's encoder presets. The default is
  HEVC on the Apple GPU on macOS and NVENC on NVIDIA.

## How the SeedVR2 hook works

Lada's `FrameRestorer` cuts each tracked mosaic into a clip of 256×256 crops (resized and
padded), restores the clip, then un-pads, resizes and mask-blends each crop back into the
frame. `SeedVR2FrameRestorer` (`video_restore/seedvr2_refiner.py`) overrides `_restore_clip`:

1. Lada BasicVSR++ restores the clip at 256 px.
2. SeedVR2 upscales and refines the restored clip to 256·s px, where s is an integer.
3. The clip's masks are resized (nearest) and its padding is multiplied by s.

Lada's own `_restore_frame` then crops out the padding, resizes to the original box and
feathers the result back in. Nothing outside the mosaic boxes is touched.

## Test

```bash
.venv/bin/python tests/test_seedvr2_hook.py   # synthetic clip through Lada + SeedVR2 + blend
```

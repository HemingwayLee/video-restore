"""vrestore: de-mosaic videos with Lada, optionally refining the restored crops with SeedVR2-3B.

Default (fast):  Lada detects mosaics, restores them with BasicVSR++, blends them back.
--seedvr2:       same, but every restored crop clip is additionally passed through
                 SeedVR2 (3B) before blending. Only the mosaic crops go through
                 SeedVR2, never the full frame.
"""

from __future__ import annotations

import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADA_WEIGHTS = os.path.join(ROOT, "models", "lada")
SEEDVR2_WEIGHTS = os.path.join(ROOT, "models", "seedvr2")
SEEDVR2_REPO = os.path.join(ROOT, "vendor", "seedvr2")

# Must be set before lada / torch are imported.
os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", LADA_WEIGHTS)
if platform.system() == "Darwin":
    # Cap MPS allocations at Metal's recommended working set (~75% of RAM) so an oversized
    # SeedVR2 batch fails with a clean "MPS out of memory" error instead of exhausting the
    # whole machine. ("0.0" would disable the limit entirely.) Override via the environment.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")

import argparse  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vrestore",
        description="Remove mosaics from video with Lada (fast). Add --seedvr2 to refine the "
                    "Lada-restored crops with SeedVR2-3B (much slower, sharper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="Input video file or directory of videos")
    p.add_argument("-o", "--output", help="Output file, or directory (default: next to input)")
    p.add_argument("--output-pattern", default=None,
                   help="Output file name pattern with {orig_file_name} "
                        "(default: {orig_file_name}.restored.mp4, or .restored-seedvr2.mp4 with --seedvr2)")
    p.add_argument("--temp-dir", default=tempfile.gettempdir(), help="Directory for temporary video files")

    g = p.add_argument_group("Lada")
    g.add_argument("--device", default=None, help="Torch device for Lada (default: cuda:0 / mps / cpu, auto)")
    g.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                   help="Half precision for Lada (default: auto)")
    g.add_argument("--detection-model", default="v4-fast", help="Lada mosaic detection model: v4-fast | v4-accurate | path")
    g.add_argument("--restoration-model", default="basicvsrpp-v1.2", help="Lada restoration model name or path")
    g.add_argument("--max-clip-length", type=int, default=180,
                   help="Max frames per mosaic clip (higher = more temporal stability, more memory)")
    g.add_argument("--detect-face-mosaics", action="store_true", help="Leave pixelated faces alone")
    g.add_argument("--encoding-preset", default=None, help="Lada encoding preset (see --list-encoding-presets)")
    g.add_argument("--encoder", default=None, help="Raw ffmpeg encoder name (overrides preset)")
    g.add_argument("--encoder-options", default="", help="Options for --encoder, e.g. '-crf 18 -preset slow'")
    g.add_argument("--list-encoding-presets", action="store_true", help="List encoding presets and exit")
    g.add_argument("--list-devices", action="store_true", help="List torch devices and exit")

    s = p.add_argument_group("SeedVR2 (optional refinement of Lada's restored crops)")
    s.add_argument("--seedvr2", action="store_true", help="Run SeedVR2 on every Lada-restored mosaic crop clip")
    s.add_argument("--seedvr2-model", default="seedvr2_ema_3b_fp16.safetensors",
                   help="SeedVR2 DiT weights (3B: seedvr2_ema_3b_fp16.safetensors, "
                        "seedvr2_ema_3b_fp8_e4m3fn.safetensors (CUDA), seedvr2_ema_3b-Q8_0.gguf, seedvr2_ema_3b-Q4_K_M.gguf)")
    s.add_argument("--seedvr2-scale", default="auto",
                   help="Upscale factor applied to Lada's 256px crops: auto | 1 | 2 | 3 | 4. "
                        "auto = enough to cover the original crop size (min 2)")
    s.add_argument("--seedvr2-max-res", type=int, default=1024, help="Cap on SeedVR2 crop resolution (px)")
    s.add_argument("--seedvr2-batch-size", type=int, default=21,
                   help="Frames per SeedVR2 batch (4n+1). Larger = better temporal consistency, more memory")
    s.add_argument("--seedvr2-color-correction", default="lab",
                   choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
    s.add_argument("--seedvr2-seed", type=int, default=42)
    s.add_argument("--seedvr2-blocks-to-swap", type=int, default=0,
                   help="Offload N DiT blocks to CPU to save VRAM (0-32; CUDA only, slower)")
    s.add_argument("--seedvr2-vae-tile", type=int, default=0,
                   help="Tiled VAE encode/decode with this tile size in px (e.g. 512) to cut peak memory; 0 = off")
    s.add_argument("--seedvr2-debug", action="store_true", help="Verbose SeedVR2 logging")
    s.add_argument("--seedvr2-download", action="store_true", help="Download SeedVR2 weights and exit")
    return p


def ensure_apple_hevc_tag(path: str) -> bool:
    """Re-tag HEVC video in an MP4/MOV from 'hev1' to 'hvc1' so QuickTime/Safari can play it.

    Lada's audio mux (ffmpeg -c copy) drops the 'hvc1' tag the encoder set, leaving ffmpeg's
    default 'hev1', which Apple players refuse. This is a stream-copy remux: no re-encoding.
    Returns True if the file was re-tagged.
    """
    import subprocess

    if os.path.splitext(path)[1].lower() not in (".mp4", ".m4v", ".mov"):
        return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,codec_tag_string", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.split() != ["hevc", "hev1"]:
        return False
    stem, ext = os.path.splitext(path)
    tmp = f"{stem}.retag{ext}"
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path, "-map", "0", "-c", "copy",
         "-tag:v", "hvc1", "-movflags", "+faststart", tmp],
        capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"Warning: could not re-tag HEVC as hvc1 (QuickTime may not open it): {result.stderr.strip()}")
        return False
    os.replace(tmp, path)
    return True


def resolve_encoder(args, video_utils) -> tuple[str, str]:
    if args.encoder:
        return args.encoder, args.encoder_options
    name = args.encoding_preset or video_utils.get_default_preset_name()
    for preset in video_utils.get_encoding_presets():
        if preset.name == name:
            return preset.encoder_name, preset.encoder_options
    sys.exit(f"Unknown encoding preset '{name}'. Use --list-encoding-presets.")


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    # Treat SIGTERM / SIGHUP (kill, closed terminal) like Ctrl-C so the restorer threads are
    # stopped, the temp file is removed and atexit cleanup runs. Without this the process dies
    # immediately and multiprocessing's resource_tracker warns about tqdm's leaked semaphore.
    import signal
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    args = build_parser().parse_args(argv)

    import torch
    from lada import ModelFiles
    from lada.cli import utils as lada_cli_utils
    from lada.restorationpipeline import load_models
    from lada.restorationpipeline.frame_restorer import FrameRestorer
    from lada.utils import audio_utils, video_utils
    from lada.utils.os_utils import get_default_torch_device, gpu_has_fp16_acceleration
    from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

    if args.list_encoding_presets:
        lada_cli_utils.dump_available_encoding_presets()
        return
    if args.list_devices:
        lada_cli_utils.dump_torch_devices()
        return

    refiner = None
    if args.seedvr2 or args.seedvr2_download:
        from video_restore.seedvr2_refiner import SeedVR2Options, SeedVR2Refiner
        if args.seedvr2_scale != "auto" and not args.seedvr2_scale.isdigit():
            sys.exit("--seedvr2-scale must be 'auto' or a positive integer")
        refiner = SeedVR2Refiner(SeedVR2Options(
            repo_dir=SEEDVR2_REPO, model_dir=SEEDVR2_WEIGHTS, dit_model=args.seedvr2_model,
            batch_size=args.seedvr2_batch_size, color_correction=args.seedvr2_color_correction,
            seed=args.seedvr2_seed, blocks_to_swap=args.seedvr2_blocks_to_swap,
            vae_tile_size=args.seedvr2_vae_tile, debug=args.seedvr2_debug))
        refiner.download_models()
        if args.seedvr2_download:
            print(f"SeedVR2 weights ready in {SEEDVR2_WEIGHTS}")
            return

    if not args.input:
        build_parser().print_help()
        sys.exit(2)
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        sys.exit(f"No such file or directory: {args.input}")

    device = torch.device(args.device or get_default_torch_device())
    fp16 = gpu_has_fp16_acceleration() if args.fp16 is None else args.fp16

    det = ModelFiles.get_detection_model_by_name(args.detection_model)
    det_path = det.path if det else args.detection_model
    if not os.path.isfile(det_path):
        sys.exit(f"Detection model '{args.detection_model}' not found (looked in {LADA_WEIGHTS}). Run ./setup.sh")
    rest = ModelFiles.get_restoration_model_by_name(args.restoration_model)
    if rest:
        rest_name, rest_path = args.restoration_model, rest.path
    else:
        rest_name, rest_path = "basicvsrpp", args.restoration_model
    if not os.path.isfile(rest_path):
        sys.exit(f"Restoration model '{args.restoration_model}' not found (looked in {LADA_WEIGHTS}). Run ./setup.sh")

    encoder, encoder_options = resolve_encoder(args, video_utils)
    pattern = args.output_pattern or ("{orig_file_name}.restored-seedvr2.mp4" if args.seedvr2
                                      else "{orig_file_name}.restored.mp4")
    if "{orig_file_name}" not in pattern:
        sys.exit("--output-pattern must contain {orig_file_name}")
    os.makedirs(args.temp_dir, exist_ok=True)

    print(f"Lada: device={device} fp16={fp16} detection={args.detection_model} restoration={args.restoration_model}")
    if refiner:
        print(f"SeedVR2: model={args.seedvr2_model} scale={args.seedvr2_scale} max_res={args.seedvr2_max_res} "
              f"batch={args.seedvr2_batch_size}")

    detection_model, restoration_model, pad_mode = load_models(
        device, rest_name, rest_path, None, det_path, fp16, args.detect_face_mosaics)

    inputs, outputs = lada_cli_utils.setup_input_and_output_paths(args.input, args.output, pattern)

    for in_path, out_path in zip(inputs, outputs):
        print(f"\n{in_path}\n  -> {out_path}")
        meta = video_utils.get_video_meta_data(in_path)
        ctor_args = (device, in_path, args.max_clip_length, rest_name, detection_model, restoration_model, pad_mode)
        if refiner:
            from video_restore.seedvr2_refiner import SeedVR2FrameRestorer
            restorer = SeedVR2FrameRestorer(*ctor_args, refiner=refiner, scale=args.seedvr2_scale,
                                            max_crop_res=args.seedvr2_max_res)
        else:
            restorer = FrameRestorer(*ctor_args)

        stem, ext = os.path.splitext(os.path.basename(out_path))
        tmp_path = os.path.join(args.temp_dir, f"{stem}.tmp{ext}")
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        bar = lada_cli_utils.Progressbar(meta)
        ok = True
        t0 = time.time()
        try:
            restorer.start()
            bar.init()
            with video_utils.VideoWriter(tmp_path, meta.video_width, meta.video_height, meta.video_fps_exact,
                                         encoder=encoder, encoder_options=encoder_options,
                                         time_base=meta.time_base, mp4_fast_start=False) as writer:
                for elem in restorer:
                    if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                        ok = False
                        bar.error = True
                        if isinstance(elem, ErrorMarker):
                            print(f"\nError during restoration: {getattr(elem, 'stack_trace', elem)}")
                        break
                    frame, pts = elem
                    writer.write(frame, pts, bgr2rgb=True)
                    bar.update()
        except KeyboardInterrupt:
            ok, interrupted = False, True
        else:
            interrupted = False
        finally:
            restorer.stop()
            bar.close(ensure_completed_bar=ok)

        if not ok:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if interrupted:
                sys.exit("\nInterrupted.")
            sys.exit(f"Failed: {in_path}")
        audio_utils.combine_audio_video_files(meta, tmp_path, out_path)
        ensure_apple_hevc_tag(out_path)
        extra = f", {restorer.clips_refined} crop clips refined by SeedVR2" if refiner else ""
        print(f"Done in {time.time() - t0:.1f}s{extra}: {out_path}")


if __name__ == "__main__":
    main()

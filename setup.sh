#!/usr/bin/env bash
# Set up the video-restore virtualenv: Lada (always) and, optionally, SeedVR2.
#
#   ./setup.sh                 # Lada only
#   ./setup.sh --with-seedvr2  # Lada + SeedVR2 code/deps + SeedVR2-3B weights
#
# Environment overrides:
#   PYTHON_VERSION   Python used for .venv (default 3.12; Lada needs >= 3.12)
#   TORCH_EXTRA      Lada torch extra: cpu | nvidia | nvidia-legacy | intel
#                    (default: cpu on macOS (includes MPS/Metal), nvidia if nvidia-smi exists)
#   SEEDVR2_MODEL    SeedVR2 DiT weights to pre-download (default seedvr2_ema_3b_fp16.safetensors)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
VENDOR="$ROOT/vendor"
MODELS="$ROOT/models"

# Pinned upstream revisions (the SeedVR2 integration hooks into Lada internals).
LADA_REPO="https://codeberg.org/ladaapp/lada.git"
LADA_REF="20cb34a20a83c72c87a991d2c949032c70085b16"
SEEDVR2_REPO="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_REF="4490bd1f482e026674543386bb2a4d176da245b9"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SEEDVR2_MODEL="${SEEDVR2_MODEL:-seedvr2_ema_3b_fp16.safetensors}"

WITH_SEEDVR2=0
for arg in "$@"; do
  case "$arg" in
    --with-seedvr2) WITH_SEEDVR2=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- system deps
need=()
command -v git     >/dev/null || need+=(git)
command -v ffmpeg  >/dev/null || need+=(ffmpeg)
command -v ffprobe >/dev/null || need+=(ffmpeg)
command -v uv      >/dev/null || need+=(uv)
if ((${#need[@]})); then
  # shellcheck disable=SC2207
  need=($(printf '%s\n' "${need[@]}" | sort -u))
  if command -v brew >/dev/null; then
    log "Installing missing tools with Homebrew: ${need[*]}"
    brew install "${need[@]}"
  else
    echo "Missing required tools: ${need[*]}. Install them and re-run." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------- helpers
fetch_repo() { # url ref dest
  local url="$1" ref="$2" dest="$3"
  if [[ -d "$dest/.git" ]] && [[ "$(git -C "$dest" rev-parse HEAD)" == "$ref" ]]; then
    return
  fi
  log "Fetching $url @ ${ref:0:10}"
  rm -rf "$dest"
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote add origin "$url"
  git -C "$dest" fetch -q --depth 1 origin "$ref"
  git -C "$dest" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }

download_verified() { # url dest expected_sha
  local url="$1" dest="$2" sha="$3"
  if [[ -f "$dest" ]] && [[ "$(sha256_of "$dest")" == "$sha" ]]; then
    echo "  ok  $(basename "$dest")"
    return
  fi
  echo "  get $(basename "$dest")"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  local got; got="$(sha256_of "$dest.part")"
  if [[ "$got" != "$sha" ]]; then
    rm -f "$dest.part"
    echo "Checksum mismatch for $(basename "$dest"): expected $sha, got $got" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

# ---------------------------------------------------------------- Lada
fetch_repo "$LADA_REPO" "$LADA_REF" "$VENDOR/lada"

if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating virtualenv ($PYTHON_VERSION) at .venv"
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
PY="$VENV/bin/python"

if [[ -z "${TORCH_EXTRA:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then TORCH_EXTRA=cpu      # macOS wheels include MPS
  elif command -v nvidia-smi >/dev/null; then TORCH_EXTRA=nvidia
  else TORCH_EXTRA=cpu; fi
fi

log "Installing Lada (+torch extra: $TORCH_EXTRA)"
# Constrain every dependency to Lada's lockfile (newer PyAV etc. break Lada).
CONSTRAINTS="$VENV/lada-constraints.txt"
uv export --project "$VENDOR/lada" --frozen --no-hashes --no-emit-project --no-header \
  --extra "$TORCH_EXTRA" -o "$CONSTRAINTS" >/dev/null
uv pip install --python "$PY" -c "$CONSTRAINTS" "$VENDOR/lada[$TORCH_EXTRA]"

log "Applying Lada's dependency patches"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for p in increase_mms_time_limit.patch remove_ultralytics_telemetry.patch fix_loading_mmengine_weights_on_torch26_and_higher.diff; do
  if patch -s -p1 -N --dry-run -d "$SITE" < "$VENDOR/lada/patches/$p" >/dev/null 2>&1; then
    patch -s -p1 -N -d "$SITE" < "$VENDOR/lada/patches/$p"
    echo "  applied $p"
  elif patch -s -p1 -R --dry-run -d "$SITE" < "$VENDOR/lada/patches/$p" >/dev/null 2>&1; then
    echo "  skip    $p (already applied)"
  else
    echo "Patch $p does not apply to the installed packages" >&2
    exit 1
  fi
done

log "Downloading Lada model weights to models/lada"
HF="https://huggingface.co/ladaapp/lada/resolve/main"
for f in lada_mosaic_detection_model_v4_fast.pt \
         lada_mosaic_detection_model_v4_accurate.pt \
         lada_mosaic_restoration_model_generic_v1.2.pth; do
  sha="$(awk -v f="$f" '$2==f {print $1}' "$VENDOR/lada/model_weights/checksums_sha256.txt")"
  download_verified "$HF/$f?download=true" "$MODELS/lada/$f" "$sha"
done

# ---------------------------------------------------------------- SeedVR2 (optional)
if ((WITH_SEEDVR2)); then
  fetch_repo "$SEEDVR2_REPO" "$SEEDVR2_REF" "$VENDOR/seedvr2"
  log "Installing SeedVR2 dependencies"
  # Keep Lada's pinned opencv / torch; only add what SeedVR2 needs on top.
  OPENCV_PIN="$("$PY" -c 'import importlib.metadata as m; print(m.version("opencv-python"))')"
  TORCH_PIN="$("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))')"
  TV_PIN="$("$PY" -c 'import importlib.metadata as m; print(m.version("torchvision"))')"
  uv pip install --python "$PY" -r "$VENDOR/seedvr2/requirements.txt" \
    "opencv-python==$OPENCV_PIN" "torch==$TORCH_PIN" "torchvision==$TV_PIN"

  log "Downloading SeedVR2 weights ($SEEDVR2_MODEL + VAE) to models/seedvr2"
  "$ROOT/vrestore" --seedvr2-download --seedvr2-model "$SEEDVR2_MODEL"
fi

log "Done."
echo "Run:  ./vrestore INPUT.mp4                # Lada only (fast)"
((WITH_SEEDVR2)) && echo "      ./vrestore INPUT.mp4 --seedvr2      # Lada + SeedVR2-3B on restored crops"
exit 0

#!/usr/bin/env bash
set -euo pipefail

# Resolve the project root from the script location, not from $PWD, so the script
# works no matter which directory it is launched from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
CONFIG_DIR="$PROJECT_ROOT/configs"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 VIDEO [OUTPUT]" >&2
  echo "       USE_EMBEDDED_CONFIG=1 $0 VIDEO [OUTPUT]   # use configs baked into the image" >&2
  exit 2
fi

VIDEO="$1"
OUTPUT="${2:-$PWD/data/output}"
IMAGE="${IMAGE:-boxing-tracking:runtime}"
USE_EMBEDDED_CONFIG="${USE_EMBEDDED_CONFIG:-0}"

[[ -f "$VIDEO" ]] || {
  echo "ERROR: video does not exist: $VIDEO" >&2
  exit 1
}

mkdir -p "$OUTPUT"
VIDEO="$(realpath "$VIDEO")"
OUTPUT="$(realpath "$OUTPUT")"

# Mounts shared by both modes.
RUN_ARGS=(
  run --rm --gpus all
  --shm-size=2g
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -v "$VIDEO:/data/input/video.mp4:ro"
  -v "$OUTPUT:/data/output"
)

if [[ "$USE_EMBEDDED_CONFIG" == "1" ]]; then
  # Embedded mode: keep the configs that were copied into the image at build time.
  echo "Using configs embedded in Docker image."
else
  # Live mode (default): mount the local configs/ folder so YAML edits take effect
  # on the next run without rebuilding the image. infer_tracks.py reads
  # /app/configs/infer_tracks.yaml, so the Docker-specific YAML (container paths)
  # is mounted on top under that name; the host's infer_tracks.yaml (local paths)
  # stays available as infer_tracks.docker.yaml's sibling but is not used.
  REQUIRED_CONFIGS=(
    infer_tracks.docker.yaml
    tracking.yaml
    birth_manager.yaml
    shot_boundary.yaml
  )
  for cfg in "${REQUIRED_CONFIGS[@]}"; do
    [[ -f "$CONFIG_DIR/$cfg" ]] || {
      echo "ERROR: required config not found: $CONFIG_DIR/$cfg" >&2
      exit 1
    }
  done

  # Optional host-side YAML validation. Skip quietly if PyYAML is unavailable;
  # the container performs its own loading at run time.
  if python3 -c "import yaml" >/dev/null 2>&1; then
    for cfg in "${REQUIRED_CONFIGS[@]}"; do
      python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$CONFIG_DIR/$cfg" || {
        echo "ERROR: invalid YAML: $CONFIG_DIR/$cfg" >&2
        exit 1
      }
    done
  else
    echo "NOTE: PyYAML not found on host; skipping local YAML validation." >&2
  fi

  RUN_ARGS+=(
    -v "$CONFIG_DIR:/app/configs:ro"
    -v "$CONFIG_DIR/infer_tracks.docker.yaml:/app/configs/infer_tracks.yaml:ro"
  )

  echo "Using live configs from: $CONFIG_DIR"
  echo "Main config: infer_tracks.docker.yaml"
  echo "Mounted as: /app/configs/infer_tracks.yaml"
  echo "Config changes do not require rebuilding the image."
fi

RUN_ARGS+=("$IMAGE")

"${DOCKER[@]}" "${RUN_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Build the Boxing Tracking runtime images.
#
#   build-runtime.sh            -> builds both:  boxing-tracking:runtime-dev
#                                                boxing-tracking:runtime
#   build-runtime.sh development -> builds only: boxing-tracking:runtime-dev
#   build-runtime.sh release     -> builds only: boxing-tracking:runtime
#
# Env overrides: CUDA_ARCH_BIN (default 89), DEV_IMAGE, RELEASE_IMAGE.

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

MODE="${1:-all}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-89}"
DEV_IMAGE="${DEV_IMAGE:-boxing-tracking:runtime-dev}"
RELEASE_IMAGE="${RELEASE_IMAGE:-boxing-tracking:runtime}"

# Always run from the repository root so the build context is correct.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

build_target() {
  local target="$1" image="$2"
  printf '\n=== Building target: %s -> %s (CUDA_ARCH_BIN=%s) ===\n' \
    "$target" "$image" "$CUDA_ARCH_BIN"
  "${DOCKER[@]}" build \
    --progress=plain \
    --target "$target" \
    --build-arg "CUDA_ARCH_BIN=$CUDA_ARCH_BIN" \
    -f docker/Dockerfile.runtime \
    -t "$image" \
    .
}

check_openpose_model() {
  test -s docker/openpose_models/pose/body_25/pose_iter_584000.caffemodel || {
    echo "ERROR: BODY_25 model docker/openpose_models/pose/body_25/pose_iter_584000.caffemodel is missing." >&2
    exit 1
  }
}

# The release stage bakes this exact ONNX from the artifacts tree.
APPEARANCE_ONNX="artifacts/models/apperance_cnn/osnet_1_x_ain(26_03)_new_stage_2.onnx"

check_release_assets() {
  test -s "$APPEARANCE_ONNX" || {
    echo "ERROR: appearance model is missing or empty: $APPEARANCE_ONNX" >&2
    exit 1
  }
  grep -q '/app/models/appearance.onnx' configs/infer_tracks.docker.yaml || {
    echo "ERROR: infer_tracks.docker.yaml must point to /app/models/appearance.onnx." >&2
    exit 1
  }
}

check_openpose_model

case "$MODE" in
  development|dev)
    build_target development "$DEV_IMAGE"
    ;;
  release)
    check_release_assets
    build_target release "$RELEASE_IMAGE"
    ;;
  all)
    check_release_assets
    build_target development "$DEV_IMAGE"
    build_target release "$RELEASE_IMAGE"
    ;;
  *)
    echo "Usage: $0 [all|development|release]" >&2
    exit 2
    ;;
esac

printf '\nDone.\n'

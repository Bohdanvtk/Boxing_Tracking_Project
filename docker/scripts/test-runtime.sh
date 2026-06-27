#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the release image. Verifies GPU access, pyopenpose, the installed
# project, core Python libraries, all YAML configs, the BODY_25 model files, the
# bundled ONNX model and the required OpenPose/Caffe shared libraries.

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

IMAGE="${IMAGE:-boxing-tracking:runtime}"

"${DOCKER[@]}" run --rm --gpus all \
  --entrypoint bash \
  "$IMAGE" \
  -lc '
    set -euo pipefail

    echo "== GPU =="
    nvidia-smi

    echo "== Python imports =="
    python3 - <<"PY"
import cv2
import numpy
import onnxruntime
import yaml
import boxing_project
from openpose import pyopenpose

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("PyYAML:", yaml.__version__)
print("boxing_project:", boxing_project.__file__)
print("pyopenpose:", pyopenpose.__file__)

# All YAML configs must parse.
for name in ("infer_tracks", "tracking", "birth_manager", "shot_boundary"):
    with open(f"/app/configs/{name}.yaml") as f:
        yaml.safe_load(f)
    print("YAML OK:", name)
print("RUNTIME IMPORTS OK")
PY

    echo "== BODY_25 model =="
    test -s /opt/openpose/models/pose/body_25/pose_deploy.prototxt
    test -s /opt/openpose/models/pose/body_25/pose_iter_584000.caffemodel

    echo "== Bundled ONNX model =="
    test -s /app/models/appearance.onnx

    echo "== Config files =="
    test -f /app/configs/infer_tracks.yaml
    test -f /app/configs/tracking.yaml
    test -f /app/configs/birth_manager.yaml
    test -f /app/configs/shot_boundary.yaml

    echo "== Shared libraries =="
    test -e /opt/openpose/lib/libopenpose.so
    test -e /opt/openpose/lib/libcaffe.so
    PYOP="$(python3 -c "from openpose import pyopenpose; print(pyopenpose.__file__)")"
    # Fail if any shared dependency of pyopenpose is unresolved.
    if ldd "$PYOP" | grep -i "not found"; then
      echo "ERROR: unresolved shared libraries for pyopenpose" >&2
      exit 1
    fi
    ldd /opt/openpose/lib/libopenpose.so | grep -i "not found" && {
      echo "ERROR: unresolved shared libraries for libopenpose.so" >&2
      exit 1
    } || true

    echo "RUNTIME FILES OK"
  '

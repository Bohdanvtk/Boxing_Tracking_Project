# Prepared multi-stage runtime image

This bundle is designed to be copied into the root of
`Boxing_Tracking_Project`.

## What is already decided

- OpenPose is compiled in a disposable builder stage.
- The final image starts from the CUDA 11.8 + cuDNN runtime image.
- Only OpenPose shared libraries, `pyopenpose`, BODY_25 and application files
  are copied into the runtime stage.
- The final release image bundles the appearance ONNX model.
- Video and output stay outside the image and are mounted at runtime.
- All four configs must exist: infer, tracking, birth_manager and shot_boundary.

## Files to copy into the repository

```text
docker/Dockerfile.runtime
docker/Dockerfile.runtime.dockerignore
docker/models/README.md
docker/scripts/build-runtime.sh
docker/scripts/test-runtime.sh
docker/scripts/run-runtime.sh
```

Keep the existing `docker/constraints.txt` with:

```text
numpy==1.23.5
opencv-python==4.8.1.78
```

Make scripts executable:

```bash
chmod +x docker/scripts/*.sh
```

## Build now, without the final ONNX model

```bash
docker/scripts/build-runtime.sh development
```

This validates the OpenPose/runtime design but does not create a publishable
release image.

## Before the final release build

1. Finish the Python code and YAML structure.
2. Put the final ONNX model at:

```text
docker/models/appearance.onnx
```

3. Set this in `configs/infer_tracks.docker.yaml`:

```yaml
tracking:
  apperance_embedding_model_path: "/app/models/appearance.onnx"
```

4. Ensure these files exist:

```text
configs/infer_tracks.docker.yaml
configs/tracking.yaml
configs/birth_manager.yaml
configs/shot_boundary.yaml
```

5. Build the release image:

```bash
docker/scripts/build-runtime.sh release
```

## Test and run

```bash
docker/scripts/test-runtime.sh

docker/scripts/run-runtime.sh /absolute/path/to/video.mp4
```

## GPU portability

The default build is optimized for the current RTX 4060 (`sm_89`). For a more
portable public image, build with several architectures, for example:

```bash
CUDA_ARCH_BIN='75;80;86;89' docker/scripts/build-runtime.sh release
```

This increases OpenPose build time and image size slightly, but supports more
NVIDIA GPU generations.

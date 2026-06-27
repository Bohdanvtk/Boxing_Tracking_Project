# Docker runtime guide

This guide covers everything about the Dockerized inference runtime: how to run
the **published image**, how to **build it yourself**, and how to **publish** a
new image to the GitHub Container Registry (GHCR).

The runtime packages OpenPose (BODY_25), the appearance/ReID ONNX model and the
full tracking pipeline into a single CUDA image, so no native OpenPose/CUDA setup
is needed on the host.

---

## How the image is built

`docker/Dockerfile.runtime` is a single multi-stage build:

```text
openpose-builder  compiles OpenPose + pyopenpose (CUDA devel base, discarded)
runtime-base      small CUDA runtime base + OpenPose shared libs + BODY_25 model
development        + Python deps + project source/configs    -> boxing-tracking:runtime-dev
release            + appearance.onnx baked in                 -> boxing-tracking:runtime
```

The heavy compiler/build tree lives only in `openpose-builder` and is **not**
copied into the final image, so the published `release` image stays as small as a
CUDA + OpenPose image can reasonably be (~10 GB uncompressed, ~3 GB pushed).

Models are handled like this:

```text
OpenPose BODY_25   docker/openpose_models/pose/body_25/pose_iter_584000.caffemodel
                   (100 MB, gitignored — see "Models required to build")
Appearance / ReID  artifacts/models/apperance_cnn/osnet_1_x_ain(26_03)_new_stage_2.onnx
                   baked into the release image at /app/models/appearance.onnx
```

Video and output never live in the image — they are mounted at run time.

---

## Prerequisites (host)

To run the image the host needs an NVIDIA GPU with:

- a recent **NVIDIA driver**;
- the **NVIDIA Container Toolkit** (`nvidia-container-toolkit`), which enables
  `docker run --gpus all`.

Quick check:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, the runtime image will work.

**Supported GPUs.** The published image is built for compute capabilities
`75;80;86;89`, i.e. Turing (RTX 20xx), Ampere (RTX 30xx / A100) and Ada
(RTX 40xx). For other architectures, build the image yourself with a matching
`CUDA_ARCH_BIN` (see below).

---

## Use the published image (pull & run)

This is the easiest path — no build, no model downloads.

```bash
docker pull ghcr.io/bohdanvtk/boxing-tracking:runtime

docker run --rm --gpus all --shm-size=2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /absolute/path/to/video.mp4:/data/input/video.mp4:ro \
  -v /absolute/path/to/output:/data/output \
  ghcr.io/bohdanvtk/boxing-tracking:runtime
```

- The input video is mounted read-only at `/data/input/video.mp4`.
- Results are written to whatever host directory you mount at `/data/output`.
- The pipeline uses the configs **baked into the image**, so no local `configs/`
  folder is required.

The dataset and tracking results appear under your mounted output directory (see
the main `README.md` → *Dataset Output* for the file layout).

---

## Build it yourself (developers)

### Models required to build

The appearance ONNX is in the repository, but the OpenPose weights are not (they
are large binaries). Before a `release` build, make sure both exist locally:

```text
docker/openpose_models/pose/body_25/pose_iter_584000.caffemodel   (100 MB, you provide)
artifacts/models/apperance_cnn/osnet_1_x_ain(26_03)_new_stage_2.onnx  (in git)
```

`pose_iter_584000.caffemodel` is the standard OpenPose BODY_25 model. Obtain it
from your existing OpenPose installation (`models/pose/body_25/`) or the OpenPose
project, and place it at the path above.

### Build

```bash
chmod +x docker/scripts/*.sh

docker/scripts/build-runtime.sh development   # boxing-tracking:runtime-dev (no ONNX baked)
docker/scripts/build-runtime.sh release       # boxing-tracking:runtime (publishable)
docker/scripts/build-runtime.sh               # both
```

`development` validates the OpenPose/runtime layers without bundling the ONNX
model; `release` produces the final, publishable image.

### GPU portability (multi-arch)

The default build targets only the local RTX 4060 (`sm_89`). For a portable
public image, build for several architectures and add a PTX fallback:

```bash
CUDA_ARCH_BIN='89;86;80;75' CUDA_ARCH_PTX='75' docker/scripts/build-runtime.sh release
```

This is exactly how the **published** image is produced, and it lets the same
image run on Turing → Ada GPUs and the A100.

Why the specific order and the PTX value matter:

- OpenPose's bundled **Caffe only compiles native code for the FIRST**
  `CUDA_ARCH_BIN` value, so the GPU you actually run on must be listed first
  (here `89` for the RTX 4060). Listing it later silently produces an image that
  fails at run time with `no kernel image is available for execution on the device`.
- `CUDA_ARCH_PTX='75'` embeds forward-compatible PTX that the driver
  JIT-compiles for any GPU with compute capability ≥ 7.5 (Turing and newer),
  which is how other GPUs are supported from a single build.

This setting only affects the OpenPose/Caffe compilation; everything else is
identical.

### Test

```bash
docker/scripts/test-runtime.sh
```

Smoke-tests GPU access, Python imports (incl. `pyopenpose`), config parsing and
that the bundled models are present.

---

## Run a local build

```bash
docker/scripts/run-runtime.sh VIDEO [OUTPUT]
```

The project root is resolved from the script location, so it can be launched from
any directory. `OUTPUT` defaults to `./data/output`.

### Live configs (default)

By default `run-runtime.sh` mounts the local `configs/` folder into the
container, so YAML edits take effect on the next run without a rebuild:

- `configs/` → `/app/configs` (read-only)
- `configs/infer_tracks.docker.yaml` → `/app/configs/infer_tracks.yaml`
  (the script reads this file; the Docker YAML uses container paths, while the
  host `infer_tracks.yaml` keeps local `/home/...` paths).

### Embedded configs

To use the configs baked into the image instead of the local ones (this is also
what end users of the published image get):

```bash
USE_EMBEDDED_CONFIG=1 docker/scripts/run-runtime.sh VIDEO [OUTPUT]
```

### Run the published image through the script

`run-runtime.sh` honours an `IMAGE` override, so it also drives the GHCR image:

```bash
IMAGE=ghcr.io/bohdanvtk/boxing-tracking:runtime \
USE_EMBEDDED_CONFIG=1 \
docker/scripts/run-runtime.sh VIDEO [OUTPUT]
```

### When is a rebuild needed?

```text
YAML changes (configs/*.yaml)                       -> rebuild NOT needed (live mode)
Python / Dockerfile / dependencies / models changes -> rebuild needed
```

A change to `run-runtime.sh` itself never needs a rebuild — it is a host script
and is not part of the image.

---

## Publish a new image to GHCR (maintainers)

The release image is pushed manually from a machine that has it built locally.

1. **Build the multi-arch release** (see above):

   ```bash
   CUDA_ARCH_BIN='89;86;80;75' CUDA_ARCH_PTX='75' docker/scripts/build-runtime.sh release
   ```

2. **Authenticate to GHCR** with a Personal Access Token that has the
   `write:packages` scope (GitHub → Settings → Developer settings → PAT):

   ```bash
   echo "<PAT>" | docker login ghcr.io -u Bohdanvtk --password-stdin
   ```

3. **Tag and push** (the namespace must be lowercase):

   ```bash
   docker tag boxing-tracking:runtime ghcr.io/bohdanvtk/boxing-tracking:runtime
   docker tag boxing-tracking:runtime ghcr.io/bohdanvtk/boxing-tracking:v0.1.0
   docker push ghcr.io/bohdanvtk/boxing-tracking:runtime
   docker push ghcr.io/bohdanvtk/boxing-tracking:v0.1.0
   ```

4. **Make the package public** so others can pull without authenticating:
   GitHub → your profile → *Packages* → `boxing-tracking` → *Package settings* →
   *Change visibility* → **Public**, and *Connect repository* →
   `Boxing_Tracking_Project`.

After that, anyone can `docker pull ghcr.io/bohdanvtk/boxing-tracking:runtime`
and run it as shown in *Use the published image* above.

# Boxing-Specific Multi-Object Tracking

A computer vision pipeline for tracking boxers through fast motion, close-range exchanges, missed detections, and broadcast camera cuts.

This project was built to solve a practical problem in boxing analysis: before you can classify punches or count boxer-specific actions, you first need stable identities over time.

<p align="center">
  <img src="assets/demo/tracking_demo.gif" alt="Boxing tracking demo" width="900" />
</p>

## Why This Project Exists

Pose detectors can find people and keypoints on individual frames, but they do not reliably preserve identity over time.

In boxing, that becomes a major problem:

- fighters move quickly;
- they overlap often;
- body parts disappear during clinches and close exchanges;
- detections can be noisy or missing;
- the broadcast frequently cuts to a different camera angle.

Without a tracker, the same temporal sequence can accidentally combine skeletons from different athletes. That makes punch classification, punch counting, and per-boxer analysis unreliable.

This project solves that identity problem first.

<p align="center">
  <img src="assets/readme/tracking_overview.jpg" alt="Stable boxer tracking overview" width="900" />
</p>

## What The Tracker Does

The tracker combines three types of evidence when deciding which detection belongs to which boxer:

- **motion** — whether a detection agrees with the predicted track position;
- **pose** — whether the current skeleton matches the previous body configuration;
- **appearance** — whether the boxer still looks visually similar.

That makes the system more robust than a simple frame-to-frame skeleton comparison.

<p align="center">
  <img src="assets/readme/multi_cue_matching.png" alt="Multi-cue matching with motion pose and appearance" width="860" />
</p>

Pose comparison is treated as useful, but not fully trustworthy on its own, because boxing posture changes quickly and some joints may be partially hidden.

<p align="center">
  <img src="assets/readme/pose_comparison.png" alt="Pose comparison between track and candidate detection" width="860" />
</p>

The appearance representation is also boxing-specific. Instead of relying only on a generic body embedding, the tracker can combine:

- body appearance;
- left glove color information;
- right glove color information;
- shorts color information.

This makes it easier to distinguish visually similar fighters in difficult scenes.

## Why Overlap Handling Matters

Boxers often move into close contact. In these moments, a normal tracker can easily corrupt identity memory because the crop or skeleton may contain mixed information from both athletes.

To reduce that risk, this tracker includes overlap-aware logic such as:

- adaptive overlap thresholds;
- partial track updates;
- temporarily blocked appearance updates;
- freeze logic for risky identity states.

<p align="center">
  <img src="assets/readme/overlap_visualization.png" alt="Overlap-aware tracking and partial updates" width="900" />
</p>

The system also uses a temporary buffer for medium-confidence appearance observations. That allows the tracker to save potentially useful visual evidence without immediately contaminating the main identity representation.

<p align="center">
  <img src="assets/readme/buffer_visualization.png" alt="Appearance recovery buffer visualization" width="900" />
</p>

## Why New Tracks Are Created Conservatively

Not every unmatched detection should immediately become a confirmed track.

In fast boxing footage, noisy detections can appear briefly and disappear again. To avoid creating unstable identities, the tracker uses a conservative birth process.

<p align="center">
  <img src="assets/readme/pending_tracks.png" alt="Pending track candidate lifecycle" width="900" />
</p>

A new candidate must survive long enough and collect enough evidence before it becomes a stable confirmed track.

This helps reduce duplicate tracks and short-lived false identities.

## Local Tracks And Global Boxer Identities

The project separates identity handling into two levels:

- **local tracks** preserve identity inside one continuous camera segment;
- **global identities** merge compatible local fragments across different camera shots.

This distinction matters because a broadcast camera cut can completely change viewpoint, scale, and visible body regions.

### Local Tracking Inside One Camera Segment

Inside a single shot, the tracker maintains local track IDs frame by frame.

<p align="center">
  <img src="assets/readme/local_tracking_difference.png" alt="Local tracking example in one camera segment" width="445" />
  <img src="assets/readme/local_tracking_difference_2.png" alt="Local tracking example in another camera segment" width="445" />
</p>

After a camera cut, the same boxer may receive a different **local ID**, because local tracking restarts in a new shot context.

### Global Identity Recovery Across Camera Cuts

After local tracking is finished, the system compares track fragments across different epochs and groups compatible fragments into a shared global boxer identity.

<p align="center">
  <img src="assets/readme/global_tracking_difference.jpg" alt="One fragment of a boxer assigned to a global identity" width="445" />
  <img src="assets/readme/global_tracking_difference_2.jpg" alt="Another fragment of the same boxer with a different local ID but the same global identity" width="445" />
</p>

These two fragments can look different because of viewpoint changes, but they still belong to the same boxer. That is why the project uses **local IDs** and **global IDs** separately.

The final result is a scene where multiple local fragments can be consolidated into stable global boxer identities.

<p align="center">
  <img src="assets/readme/global_tracking.jpg" alt="Global boxer identities after clustering" width="860" />
</p>

For this step, stable local fragments are compared using track-level appearance representations, and conservative clustering is used to recover cross-shot identity continuity.

<p align="center">
  <img src="assets/readme/global_similarity_matrix.png" alt="Global similarity ranking and clustering evidence" width="980" />
</p>

## Why This Project Is Useful

This tracker is not only a visualization tool.

It is meant to be a foundation for downstream boxing-analysis tasks such as:

- punch classification;
- punch counting per boxer;
- temporal action recognition;
- boxer-specific movement analysis;
- later dataset preparation for learning-based models.

In other words, the project solves a prerequisite problem: it makes boxer-centered temporal analysis possible.

## Quick Start

### Installation

Dependencies are split by use case:

```bash
# Lightweight: only read observations.parquet with the Results API
pip install -r requirements/results.txt

# Full inference pipeline (heavy Python dependencies)
pip install -r requirements/inference.txt   # or: pip install -r requirements.txt
```

The recommended way to run inference is through Docker — all heavy Python and
system dependencies (OpenPose, Caffe, CUDA/cuDNN, OpenCV, ONNX Runtime) are
installed inside the image. If you only consume the resulting
`observations.parquet`, you do not need to install the inference requirements
locally; `requirements/results.txt` (numpy, pandas, pyarrow) is enough.

### Inference

The inference pipeline is configured through:

```text
configs/infer_tracks.yaml
```

Run the tracker from the repository root:

```bash
PYTHONPATH=src python scripts/infer_tracks.py
```

Main runtime configuration is currently managed through:

```text
configs/tracking.yaml
configs/birth_manager.yaml
configs/shot_boundary.yaml
```

## Dataset Output

The final pipeline stage assembles a single public dataset file:

```text
<save_dir>/dataset/observations.parquet
```

Each row is one active local track on one frame — including frames where the
detection was missed (the track stays alive with its predicted geometry but
without a detection payload). Geometry (`bbox_*`, `center_*`) always comes from
the track state; large appearance embeddings and internal/debug fields are not
exported (only `has_*` availability flags and `e_app_coverage` are kept).
`global_track_id` is `null` for local tracks without a confident global
assignment.

Reading it back:

```python
import pandas as pd
obs = pd.read_parquet("data/output/test/dataset/observations.parquet")

# a specific local track on a specific frame
row = obs[(obs.epoch_id == 6) & (obs.local_track_id == 2) & (obs.frame_idx == 348)]
bbox = row.iloc[0][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].tolist()

# all observations of a global boxer
boxer = obs[obs.global_track_id == 1].sort_values(["epoch_id", "frame_idx"])

# a frame range
segment = obs[(obs.global_track_id == 1) & obs.frame_idx.between(300, 400)]
```

### Results API

For convenience, `boxing_project.results` is a small read-only wrapper (numpy +
pandas only) that turns the parquet into ordered per-boxer sequences ready for
downstream models:

```python
from boxing_project.results import BoxingResults

results = BoxingResults("data/output/test")  # or .../dataset/observations.parquet

segment = (
    results
    .global_id(1)
    .epoch(6)
    .window(start_frame=444, length=20)
)

model_input = segment.kps            # (20, 25, 3)  -> [x, y, confidence]
model_mask = segment.detection_mask  # (20,)        -> True where a real detection exists
```

`window(start_frame, length)` always returns exactly `length` time positions,
padding frames with no observation; `frames(start, end)` returns only the rows
that really exist in the inclusive range. Shapes:

```python
segment.frames.shape            # (20,)
segment.bbox.shape              # (20, 4)
segment.kps.shape               # (20, 25, 3)
segment.observation_mask.shape  # (20,)
segment.detection_mask.shape    # (20,)
```

A selection that spans several boxers is split with `selection.segments()`,
which returns a `SegmentCollection` keyed by global id.

## Current Limitations

- difficult long overlaps can still cause identity errors;
- tracking quality depends on detection and keypoint quality;
- some global matches remain ambiguous after severe viewpoint changes;
- the current implementation is a research-oriented engineering prototype.

## Future Work

- improve the public inference API;
- add quantitative tracking evaluation;
- extend global identity recovery across harder camera switches;
- integrate downstream punch-classification models.
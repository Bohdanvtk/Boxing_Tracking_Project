# Appearance ReID Fine-Tuning

Part of the [Boxing-Specific Multi-Object Tracking](README.md) project.

The tracker uses an appearance embedding to compare the visual identity of a
current detection with an existing track. The repository includes an optional
training workflow that was used to fine-tune an OSNet-style encoder for boxing
footage.

This training workflow is separate from normal inference. It is needed only
when creating or improving the appearance model; inference can use an existing
ONNX model directly.

## Training Dependencies

Install the additional ReID training dependencies:

```bash
python -m pip install -r requirements-reid.txt
```

These dependencies include PyTorch, Torchvision, Torchreid, ONNX, and the
supporting training libraries. They are kept separate because they are much
heavier than the normal result-processing dependencies.

## Pair Dataset Format

Training uses labelled image pairs:

```text
dataset_root/
├── A/
│   ├── 000001__pos_easy__w_100.jpg
│   └── 000002__neg_hard__w_280.jpg
├── B/
│   ├── 000001__pos_easy__w_100.jpg
│   └── 000002__neg_hard__w_280.jpg
└── Label/
    ├── 000001.txt
    └── 000002.txt
```

The numeric prefix is the pair identifier and must match across `A`, `B`, and
`Label`.

Labels have the following meaning:

```text
1 = the two crops show the same boxer
0 = the two crops show different boxers
```

An optional filename suffix such as `__w_280` assigns a sample weight of `2.80`.
When no weight suffix is present, the default sample weight is `1.0`.

## Fine-Tuning OSNet

Example training command:

```bash
python scripts/train_reid_osnet_siamese.py \
  --train_roots data/reid/train_stage_1 data/reid/train_stage_2 \
  --val_root data/reid/validation \
  --model_name osnet_x0_5 \
  --batch_size 32 \
  --epochs_per_stage 20 \
  --lr 1e-4 \
  --out_dir artifacts/reid/osnet_x0_5
```

Each path passed to `--train_roots` is processed as a training stage. This makes
it possible to organise progressively harder or differently balanced pair
sets without mixing every sample into one directory.

The training script:

- resizes crops to the ReID input size of `256 × 128`;
- builds a shared Siamese appearance encoder;
- produces L2-normalised embeddings;
- minimises contrastive loss on positive and negative boxer pairs;
- supports per-pair sample weighting;
- applies validation-based learning-rate reduction and early stopping;
- stores stage checkpoints and a combined training history.

Typical outputs are:

```text
artifacts/reid/osnet_x0_5/
├── stage_01/
│   ├── best.pth
│   ├── last.pth
│   └── epoch_...pth
├── stage_02/
│   └── ...
├── final.pth
└── history.json
```

## Resuming Training

Training can continue from an existing stage checkpoint:

```bash
python scripts/train_reid_osnet_siamese.py \
  --train_roots data/reid/train_stage_1 data/reid/train_stage_2 \
  --val_root data/reid/validation \
  --model_name osnet_x0_5 \
  --resume_weights artifacts/reid/osnet_x0_5/stage_01/last.pth \
  --out_dir artifacts/reid/osnet_x0_5
```

The script determines the next epoch from the checkpoint directory and
continues inside the existing run directory.

## ONNX Export and Runtime Use

The tracker runtime consumes an ONNX appearance model. A PyTorch checkpoint
must therefore be exported before it is referenced by inference.

The export model architecture must be exactly the same as the architecture
used during training. The model name, projection head, input shape, and
checkpoint state dictionary must agree; otherwise strict checkpoint loading or
runtime embeddings will be invalid.

The repository includes:

```text
scripts/export_reid_onnx.py
```

At the current revision, that script constructs a fixed `TransReID`
configuration internally. An OSNet checkpoint trained with
`--model_name osnet_x0_5` must not be presented as safely exportable through
that fixed configuration. Before using the exporter for OSNet, make the export
script accept `--model_name` and construct the same `SiameseConfig` that was
used for training.

After a compatible ONNX model has been produced, set its path in:

```yaml
tracking:
  apperance_embedding_model_path: "artifacts/models/appearance/osnet_boxing.onnx"
```

inside `configs/infer_tracks.yaml`.

This separation is deliberate:

```text
ReID training     learns boxing-specific appearance embeddings
ONNX export       converts the trained encoder for deployment
tracking inference uses the exported embeddings as one identity cue
```

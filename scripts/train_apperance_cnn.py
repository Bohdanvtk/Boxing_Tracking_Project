from pathlib import Path
import yaml
import tensorflow as tf

from src.boxing_project.utils.config import set_seed
from src.boxing_project.apperance_embedding.dataset import CropPairs, prepare_contrastive_arrays
from src.boxing_project.apperance_embedding.cnn_model import AppearanceCNNConfig, build_appearance_cnn
from src.boxing_project.apperance_embedding.losses import contrastive_loss


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(project_root / "configs" / "train_apperance_cnn.yaml")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ---------- DATA ----------
    data_cfg = cfg["data"]
    train_pairs = CropPairs.from_npz(str(project_root / data_cfg["train_pairs"]))
    val_pairs   = CropPairs.from_npz(str(project_root / data_cfg["val_pairs"]))

    image_size = tuple(data_cfg.get("image_size", [128, 128]))
    to_rgb = bool(data_cfg.get("to_rgb", False))

    X1_tr, X2_tr, y_tr = prepare_contrastive_arrays(train_pairs, image_size=image_size, to_rgb=to_rgb)
    X1_va, X2_va, y_va = prepare_contrastive_arrays(val_pairs,   image_size=image_size, to_rgb=to_rgb)

    batch_size = int(cfg["training"]["batch_size"])

    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X1_tr, X2_tr), y_tr))
        .shuffle(len(y_tr), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(((X1_va, X2_va), y_va))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ---------- MODEL ----------
    model_cfg = cfg["model"]
    cnn_cfg = AppearanceCNNConfig(
        image_size=image_size,
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        backbone=str(model_cfg.get("backbone", "mobilenetv3small")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        l2_reg=float(model_cfg.get("l2_reg", 0.0)),
        train_backbone=bool(model_cfg.get("train_backbone", True)),
    )

    encoder = build_appearance_cnn(cnn_cfg)

    # Siamese training wrapper: two crops -> distance
    in_a = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="crop_a")
    in_b = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="crop_b")

    e_a = encoder(in_a)
    e_b = encoder(in_b)

    distance = tf.norm(e_a - e_b, axis=-1, keepdims=True)  # (B,1)

    siamese = tf.keras.Model([in_a, in_b], distance, name="appearance_siamese")

    # ---------- TRAIN ----------
    lr = float(cfg["training"]["learning_rate"])
    margin = float(cfg["training"].get("margin", 1.0))

    siamese.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=contrastive_loss(margin=margin),
    )

    siamese.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs"]),
    )

    # ---------- SAVE ----------
    save_cfg = cfg["training"]
    save_dir = project_root / save_cfg.get("save_dir", "artifacts/models/apperance_cnn")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / save_cfg.get("save_name", "apperance_encoder.h5")

    encoder.save(save_path)
    print(f"\nSaved appearance encoder to: {save_path}")


if __name__ == "__main__":
    main()

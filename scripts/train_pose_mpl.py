# scripts/train_pose_mlp.py

from pathlib import Path
import yaml
import tensorflow as tf

from src.boxing_project.utils.config import set_seed
from src.boxing_project.pose_embeding.mpl_model import PoseMLPConfig, build_pose_mlp
from src.boxing_project.pose_embeding.dataset import PosePairs, prepare_contrastive_arrays
from src.boxing_project.pose_embeding.losses import contrastive_loss


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    project_root = Path(__file__).resolve().parents[1]

    # ---------- CONFIG ----------
    cfg = load_yaml(project_root / "configs" / "train_pose_mpl.yaml")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ---------- DATA ----------
    data_cfg = cfg["data"]

    train_pairs = PosePairs.from_npz(data_cfg["train_pairs"])
    val_pairs   = PosePairs.from_npz(data_cfg["val_pairs"])

    num_keypoints = int(data_cfg["num_keypoints"])
    batch_size = int(cfg["training"]["batch_size"])

    X1_tr, X2_tr, y_tr = prepare_contrastive_arrays(train_pairs)
    X1_va, X2_va, y_va = prepare_contrastive_arrays(val_pairs)

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

    pose_cfg = PoseMLPConfig(
        num_keypoints=num_keypoints,
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        l2_reg=float(model_cfg.get("l2_reg", 0.0)),
    )

    # encoder (ТЕ, ЩО НАМ ПОТРІБНО В ПРОДАКШЕНІ)
    base_mlp = build_pose_mlp(pose_cfg)

    # siamese training wrapper
    input_a = tf.keras.Input(shape=(num_keypoints * 2,), name="pose_a")
    input_b = tf.keras.Input(shape=(num_keypoints * 2,), name="pose_b")

    emb_a = base_mlp(input_a)
    emb_b = base_mlp(input_b)

    distance = tf.keras.layers.Lambda(
        lambda t: tf.norm(t[0] - t[1], axis=-1, keepdims=True),
        name="l2_distance",
    )([emb_a, emb_b])

    siamese_model = tf.keras.Model(
        inputs=[input_a, input_b],
        outputs=distance,
        name="pose_siamese"
    )

    # ---------- TRAIN ----------
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(cfg["training"]["learning_rate"])
        ),
        loss=contrastive_loss(margin=1.0),
    )

    siamese_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs"]),
    )


    # ---------- SAVE ----------
    save_cfg = cfg["training"]
    save_dir = project_root / save_cfg.get("save_dir", "artifacts/models/pose_mlp")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / save_cfg.get("save_name", "pose_mlp.keras")

    # ❗ ЗБЕРІГАЄМО ТІЛЬКИ ENCODER
    base_mlp.save(save_path)
    print(f"\nSaved pose encoder to: {save_path}")


if __name__ == "__main__":
    main()

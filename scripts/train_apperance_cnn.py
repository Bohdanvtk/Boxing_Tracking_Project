from pathlib import Path
import yaml
import tensorflow as tf
from datetime import datetime

from src.boxing_project.utils.config import set_seed
from src.boxing_project.apperance_embedding.dataset import FolderPairsConfig, CropPairsFolder
from src.boxing_project.apperance_embedding.preprocessing import preprocess_crops_tf
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

    image_size = tuple(data_cfg.get("image_size", [128, 128]))
    to_rgb = bool(data_cfg.get("to_rgb", False))
    batch_size = int(cfg["training"]["batch_size"])

    # ---- helper: build dataset from one folder ----
    def build_one_ds(root_dir: Path, shuffle: bool):
        return CropPairsFolder.from_folder(
            FolderPairsConfig(root_dir=root_dir, image_size=image_size, seed=seed)
        ).as_tf_dataset(batch_size=batch_size, shuffle=shuffle)

    # ---- read roots (support both single and list) ----
    if "train_roots" in data_cfg:
        train_roots = [project_root / Path(p) for p in data_cfg["train_roots"]]
    else:
        train_roots = [project_root / Path(data_cfg["train_root"])]


    val_root = project_root / Path(data_cfg["val_root"])



    # preprocessing тепер тут, централізовано
    def preprocess_pair(batch, y):
        img_a, img_b = batch

        # preprocess_crops_tf має привести до float32 і нормалізувати як треба
        img_a = preprocess_crops_tf(img_a, image_size=image_size, to_rgb=to_rgb)
        img_b = preprocess_crops_tf(img_b, image_size=image_size, to_rgb=to_rgb)

        return (img_a, img_b), y

    # ---- build ONE val dataset (shared) ----
    val_ds = (
        build_one_ds(val_root, shuffle=False)
        .map(preprocess_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ---- build train datasets list (each fight separately) ----
    train_datasets = [
        build_one_ds(r, shuffle=True)
        .map(preprocess_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        for r in train_roots
    ]


    # ---------- MODEL ----------
    model_cfg = cfg["model"]
    cnn_cfg = AppearanceCNNConfig(
        image_size=image_size,
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        backbone=str(model_cfg.get("backbone", "mobilenetv3large")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        l2_reg=float(model_cfg.get("l2_reg", 0.0)),
        train_backbone=bool(model_cfg.get("train_backbone", True)),
    )

    encoder = build_appearance_cnn(cnn_cfg)

    in_a = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="crop_a")
    in_b = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="crop_b")

    e_a = encoder(in_a)
    e_b = encoder(in_b)
    distance = tf.keras.layers.Lambda(
        lambda t: tf.norm(t[0] - t[1], axis=-1, keepdims=True),
        name="l2_distance",
    )([e_a, e_b])

    siamese = tf.keras.Model([in_a, in_b], distance, name="appearance_siamese")

    # ---------- TRAIN ----------
    train_cfg = cfg["training"]
    lr = float(train_cfg["learning_rate"])
    margin = float(train_cfg.get("margin", 1.0))
    epochs = int(train_cfg["epochs"])

    loss_fn = contrastive_loss(margin=margin)

    for stage_idx, (train_ds, root) in enumerate(zip(train_datasets, train_roots), start=1):
        print(f"\n=== Stage {stage_idx}/{len(train_datasets)}: training on {root} ===")

        is_fight1 = "fight_1" in str(root)

        # learning rate
        lr_stage = lr * 0.3 if is_fight1 else lr

        # early stopping params
        patience_stage = 1 if is_fight1 else 3
        restore_best = False if is_fight1 else True

        siamese.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage),
            loss=loss_fn,
        )

        early_stop_stage = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stage,
            min_delta=0.0,
            restore_best_weights=restore_best,
            verbose=1,
        )

        siamese.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop_stage],
        )

    # ---------- SAVE ----------
    save_dir = project_root / Path(train_cfg.get("save_dir", "artifacts/models/apperance_cnn"))
    save_dir.mkdir(parents=True, exist_ok=True)

    date_tag = datetime.now().strftime("%m_%d")
    bb = cnn_cfg.backbone.lower()

    base_name = train_cfg.get("save_name", "apperance_encoder")
    base_name = Path(base_name).stem

    final_name = f"{base_name}_{date_tag}_{bb}.keras"
    save_path = save_dir / final_name

    encoder.save(save_path)
    print(f"\nSaved appearance encoder to: {save_path}")


if __name__ == "__main__":
    main()

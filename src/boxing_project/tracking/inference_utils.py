import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from boxing_project.tracking.tracker import openpose_people_to_detections
from boxing_project.shot_boundary.inference import ShotBoundaryInferencer, ShotBoundaryInferConfig
from boxing_project.tracking.image_utils import render_tracking_overlays, extract_boxing_crops, attach_overlap_info_to_detections, keypoints_to_intersection_bbox
import matplotlib.pyplot as plt
from boxing_project.tracking.openpose_processing import preprocess_image, set_openpose_module
from boxing_project.tracking.frame_processing import (
    prepare_frame_detections_from_keypoints,
    update_tracker_from_detections,
)



"""
This module contains reusable inference components:
- OpenPose initialization
- Image preprocessing
- Keypoint-based bbox extraction
- Frame processing (OpenPose -> tracker -> drawing)
- Visualization loop
"""

op = None  # will be set in init_openpose_from_config()


def init_openpose_from_config(openpose_cfg: dict):
    """
    Initialize OpenPose wrapper from YAML config.
    Returns (op_module, opWrapper).
    """
    global op

    root = os.path.expanduser(openpose_cfg["root"])
    py_path = os.path.join(root, "build", "python")

    if py_path not in sys.path:
        sys.path.append(py_path)

    try:
        from openpose import pyopenpose as op_module
    except Exception as e:
        raise RuntimeError(
            "Failed to import pyopenpose. Check OpenPose installation and the 'root' path."
        ) from e

    op = op_module
    set_openpose_module(op_module)

    params = dict()
    params["model_folder"] = os.path.join(root, "models")
    params["hand"] = openpose_cfg.get("hand", False)
    params["face"] = openpose_cfg.get("face", False)
    params["net_resolution"] = openpose_cfg.get("net_resolution", "-1x256")
    params["num_gpu"] = openpose_cfg.get("num_gpu", 1)
    params["num_gpu_start"] = openpose_cfg.get("num_gpu_start", 0)
    params["render_pose"] = openpose_cfg.get("render_pose", 0)
    params["disable_blending"] = openpose_cfg.get("disable_blending", True)
    params["number_people_max"] = openpose_cfg.get("number_people_max", 5)
    params["disable_multi_thread"] = openpose_cfg.get("disable_multi_thread", True)

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return op_module, opWrapper


def _clip_bbox_xyxy(bbox, img_w: int, img_h: int):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), img_w - 1))
    x2 = max(0, min(int(x2), img_w))
    y1 = max(0, min(int(y1), img_h - 1))
    y2 = max(0, min(int(y2), img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_features_with_hsv(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 32-dimensional HSV color histogram feature vector.

    Steps:
    1. Resize image to 32x32 for consistency
    2. Convert from BGR to HSV color space
    3. Compute histogram over H and S channels
    4. Normalize and flatten to a 1D feature vector
    """

    # Check for invalid input
    if image is None or image.size == 0:
        return None

    # Step 1: Resize image to fixed size (improves histogram stability)
    image = cv2.resize(image, (32, 32))

    # Step 2: Convert image from BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Compute 2D histogram for H and S channels
    # Channels:
    #   0 -> Hue (color)
    #   1 -> Saturation (color intensity)
    # We ignore V (brightness) to make features robust to lighting
    hist = cv2.calcHist(
        [hsv],          # input image
        [0, 1],         # channels: H and S
        None,           # no mask (use entire image)
        [8, 4],         # number of bins (H: 8, S: 4)
        [0, 180, 0, 256]  # value ranges for H and S
    )

    # Step 4: Normalize histogram (scale values to comparable range)
    hist = cv2.normalize(hist, hist)

    # Flatten histogram to 1D vector (shape: 32,)
    hist = hist.flatten()

    # Convert to float32 (useful for ML models)
    return hist.astype(np.float32)


def _l2_normalize_feature(vec):
    if vec is None:
        return None

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None

    return (arr / norm).astype(np.float32)


def build_fused_appearance_embedding(
    body_feat,
    left_glove_feat,
    right_glove_feat,
    shorts_feat,
    *,
    w_body: float = 1.0,
    w_left_glove: float = 0.5,
    w_right_glove: float = 0.5,
    w_shorts: float = 0.75,
    part_dim: int = 32,
) -> np.ndarray | None:
    body = _l2_normalize_feature(body_feat)
    if body is None:
        return None

    part_dim = max(0, int(part_dim))

    def _part_or_zero(part_feat):
        part = _l2_normalize_feature(part_feat)
        if part is None or part.size != part_dim:
            return np.zeros(part_dim, dtype=np.float32)
        return part

    left = _part_or_zero(left_glove_feat)
    right = _part_or_zero(right_glove_feat)
    shorts = _part_or_zero(shorts_feat)

    fused = np.concatenate(
        [
            float(w_body) * body,
            float(w_left_glove) * left,
            float(w_right_glove) * right,
            float(w_shorts) * shorts,
        ],
        axis=0,
    ).astype(np.float32)

    return _l2_normalize_feature(fused)


def build_fused_appearance_embedding_with_mask(
    body_feat,
    left_glove_feat,
    right_glove_feat,
    shorts_feat,
    *,
    w_body: float = 1.0,
    w_left_glove: float = 0.5,
    w_right_glove: float = 0.5,
    w_shorts: float = 0.75,
    part_dim: int = 32,
):
    """Build fixed-length fused appearance plus valid-part mask and coverage.

    Body appearance is required. Optional part segments are zero-filled when
    unavailable and marked invalid in the returned mask so matching treats them
    as unknown rather than different.
    """
    body = _l2_normalize_feature(body_feat)
    if body is None:
        return None, None, 0.0

    part_dim = max(0, int(part_dim))
    weights = {
        "body": float(w_body),
        "left_glove": float(w_left_glove),
        "right_glove": float(w_right_glove),
        "shorts": float(w_shorts),
    }
    total_weight = sum(max(weight, 0.0) for weight in weights.values())
    visible_weight = max(weights["body"], 0.0)

    def _part_or_zero_and_mask(part_feat, weight):
        nonlocal visible_weight
        part = _l2_normalize_feature(part_feat)
        if part is None or part.size != part_dim:
            return (
                np.zeros(part_dim, dtype=np.float32),
                np.zeros(part_dim, dtype=bool),
            )

        visible_weight += max(float(weight), 0.0)
        return part, np.ones(part_dim, dtype=bool)

    left, left_mask = _part_or_zero_and_mask(left_glove_feat, weights["left_glove"])
    right, right_mask = _part_or_zero_and_mask(right_glove_feat, weights["right_glove"])
    shorts, shorts_mask = _part_or_zero_and_mask(shorts_feat, weights["shorts"])

    fused = np.concatenate(
        [
            weights["body"] * body,
            weights["left_glove"] * left,
            weights["right_glove"] * right,
            weights["shorts"] * shorts,
        ],
        axis=0,
    ).astype(np.float32)

    e_app = _l2_normalize_feature(fused)
    if e_app is None:
        return None, None, 0.0

    e_app_valid_mask = np.concatenate(
        [
            np.ones(body.size, dtype=bool),
            left_mask,
            right_mask,
            shorts_mask,
        ],
        axis=0,
    )
    e_app_coverage = 0.0 if total_weight <= 1e-12 else visible_weight / total_weight
    e_app_coverage = float(np.clip(e_app_coverage, 0.0, 1.0))

    return e_app, e_app_valid_mask, e_app_coverage


def _kp_center(kp: np.ndarray, conf_th: float):
    valid = kp[:, 2] > conf_th
    if not np.any(valid):
        return None
    return np.mean(kp[valid, :2], axis=0)


def _center_dist(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.linalg.norm(a - b))


def _attach_center_distance_overlap_metadata(detections, cfg) -> None:
    # Geometry is attached by image_utils; tracker owns adaptive risk decisions.
    for det in detections:
        det.meta.setdefault("min_center_dist_norm", float("inf"))
        det.meta.setdefault("center_dist_norm_det_idx", None)
        det.meta.setdefault("overlap_relations", [])


def select_top_with_nearest(
    kps: np.ndarray,
    conf_th: float,
    top_count: int = 3,
    n: int = 3,
    intersect: int = 1,
) -> np.ndarray:
    """
    Keep top detections by original OpenPose order.
    Then add up to n nearest extra detections.

    intersect:
        max number of top detections that may point to the same extra detection.

    Important:
        top detections are never used as nearest extras.
    """

    total = len(kps)
    if total <= top_count:
        return kps

    top_count = min(top_count, total)
    top_indices = list(range(top_count))

    # extra candidates start only AFTER top detections
    extra_indices = list(range(top_count, total))

    centers = [_kp_center(kps[i], conf_th) for i in range(total)]

    # extra_idx -> how many top detections selected it as neighbor
    usage_count = {idx: 0 for idx in extra_indices}

    selected = list(top_indices)
    selected_extras = []

    # candidates: (distance, top_idx, extra_idx)
    pairs = []
    for top_idx in top_indices:
        for extra_idx in extra_indices:
            dist = _center_dist(centers[top_idx], centers[extra_idx])
            pairs.append((dist, top_idx, extra_idx))

    pairs.sort(key=lambda x: x[0])

    for dist, top_idx, extra_idx in pairs:
        if len(selected_extras) >= n:
            break

        if usage_count[extra_idx] >= intersect:
            continue

        # додаємо extra detection у фінальний список тільки один раз
        if extra_idx not in selected:
            selected.append(extra_idx)
            selected_extras.append(extra_idx)

        usage_count[extra_idx] += 1

    return kps[selected]

def _store_matched_crops(
    frame: np.ndarray,
    detections,
    matches,
    frame_dir: Optional[Path] = None,
) -> None:
    h, w = frame.shape[:2]
    if frame_dir is None:
        return

    base_dir = Path(frame_dir) / "extra" / "matched_local"
    base_dir.mkdir(parents=True, exist_ok=True)

    for track_id, det_idx in matches:
        if det_idx < 0 or det_idx >= len(detections):
            continue

        track_dir = base_dir / f"track_{int(track_id):03d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        det = detections[det_idx]
        raw = det.meta.get("raw", {})

        # 1. Save tight person bbox crop
        bb = _clip_bbox_xyxy(raw.get("bbox", None), img_w=w, img_h=h)
        if bb is not None:
            x1, y1, x2, y2 = bb
            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:
                out_name = f"det_{int(det_idx):03d}_person.jpg"
                cv2.imwrite(str(track_dir / out_name), crop)

        # 2. Save intersection bbox crop
        intersection_bb = _clip_bbox_xyxy(
            raw.get("bbox_for_intersection", None),
            img_w=w,
            img_h=h,
        )

        if intersection_bb is not None:
            x1, y1, x2, y2 = intersection_bb
            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:
                out_name = f"det_{int(det_idx):03d}_intersection_bbox.jpg"
                cv2.imwrite(str(track_dir / out_name), crop)

        # 3. Save part crops
        left_glove_crop = raw.get("left_glove_crop")
        right_glove_crop = raw.get("right_glove_crop")
        shorts_crop = raw.get("shorts_crop")

        if left_glove_crop is not None and left_glove_crop.size != 0:
            out_name = f"det_{int(det_idx):03d}_left_glove.jpg"
            cv2.imwrite(str(track_dir / out_name), left_glove_crop)

        if right_glove_crop is not None and right_glove_crop.size != 0:
            out_name = f"det_{int(det_idx):03d}_right_glove.jpg"
            cv2.imwrite(str(track_dir / out_name), right_glove_crop)

        if shorts_crop is not None and shorts_crop.size != 0:
            out_name = f"det_{int(det_idx):03d}_shorts.jpg"
            cv2.imwrite(str(track_dir / out_name), shorts_crop)

def _save_global_matched_crops(
    frame: np.ndarray,
    detections,
    global_matches,
    frame_dir: Optional[Path] = None,
) -> None:
    h, w = frame.shape[:2]
    if frame_dir is None:
        return

    matched_dir = Path(frame_dir) / "extra" / "matched_global"
    matched_dir.mkdir(parents=True, exist_ok=True)

    for global_id, det_idx in global_matches:
        if det_idx < 0 or det_idx >= len(detections):
            continue
        raw = detections[det_idx].meta.get("raw", {})
        bb = _clip_bbox_xyxy(raw.get("bbox", None), img_w=w, img_h=h)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out_name = f"global_id_{int(global_id):03d}_det_{int(det_idx):03d}.jpg"
        cv2.imwrite(str(matched_dir / out_name), crop)

def show_crop(ax, img, title: str):
    """
    Показує одну картинку в одному subplot.

    Parameters
    ----------
    ax : matplotlib axis
        Область, куди малюємо картинку.
    img : np.ndarray | None
        Crop-зображення або None.
    title : str
        Назва над картинкою.
    """
    # Підпис над картинкою
    ax.set_title(title)

    # Прибираємо осі, бо для картинок вони не потрібні
    ax.axis("off")

    # Якщо crop відсутній, показуємо текст "None"
    if img is None:
        ax.text(0.5, 0.5, "None", ha="center", va="center", fontsize=12)
        return

    # OpenCV читає зображення як BGR,
    # а matplotlib очікує RGB,
    # тому перед показом треба конвертувати
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Відображаємо картинку
    ax.imshow(img_rgb)

def visualize_boxing_crops(original_img, people):
    """
    Для кожної людини показує 4 зображення в одному рядку:
    1. повний кадр з bbox людини
    2. crop лівої рукавиці
    3. crop правої рукавиці
    4. crop шортів
    """
    for i, person in enumerate(people):
        # Створюємо 4 subplot-и в одному рядку
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # -------------------------------------------------
        # 1. Повне зображення + bbox людини
        # -------------------------------------------------
        full_img = original_img.copy()

        bbox = person.get("bbox", None)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)

            # Малюємо bbox людини зеленим прямокутником
            cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        show_crop(axes[0], full_img, f"Person {i} bbox")

        # -------------------------------------------------
        # 2. Ліва рукавиця
        # -------------------------------------------------
        show_crop(axes[1], person.get("left_glove_crop"), "Left glove")

        # -------------------------------------------------
        # 3. Права рукавиця
        # -------------------------------------------------
        show_crop(axes[2], person.get("right_glove_crop"), "Right glove")

        # -------------------------------------------------
        # 4. Шорти
        # -------------------------------------------------
        show_crop(axes[3], person.get("shorts_crop"), "Shorts")

        # Робимо відступи між subplot-ами красивішими
        plt.tight_layout()

        # Показуємо весь рядок картинок
        plt.show()

@dataclass
class FrameTrackingResult:
    """
    Lightweight per-frame result.
    Stores only tracking data and path to frame on disk.
    """
    frame_idx: int
    detections: list
    log: dict
    frame_path: Path


def _top_track_ids_by_hits(tracker, k: int = 4) -> set[int]:
    tracks = tracker.get_active_tracks(confirmed_only=False)

    tracks = sorted(
        tracks,
        key=lambda t: int(getattr(t, "hits", 0)),
        reverse=True,
    )

    return {int(t.track_id) for t in tracks[:k]}


def process_frame(result, tracker, original_img, conf_th, app_embedder, g: int, reset_mode: bool, frame_dir: Optional[Path] = None):
    """Backward-compatible single-frame wrapper used by legacy call sites."""
    if result.poseKeypoints is None:
        return [], {"matches": [], "cost_matrix": np.zeros((0, 0)), "active_tracks": [], "epoch_id": int(getattr(tracker, "_epoch_id", 1))}

    detections = prepare_frame_detections_from_keypoints(
        kps=result.poseKeypoints,
        original_img=original_img,
        conf_th=conf_th,
        tracker=tracker,
        app_embedder=app_embedder,
        select_top_with_nearest=select_top_with_nearest,
        extract_features_with_hsv=extract_features_with_hsv,
        build_fused_appearance_embedding_with_mask=build_fused_appearance_embedding_with_mask,
    )
    detections, log = update_tracker_from_detections(detections=detections, tracker=tracker, g=g, reset_mode=reset_mode)
    if frame_dir is not None:
        _store_matched_crops(original_img, detections, log.get("matches", []), frame_dir=frame_dir)
    return detections, log

def visualize_sequence(opWrapper, tracker, app_emb_path, sb_cfg: dict, images, save_width, merge_n,
                    save_dir: Path | None,
                    graph_clustering_params: dict | None = None,
                    pipeline_cfg: dict | None = None):
    from boxing_project.apperance_embedding.inference import AppearanceEmbedder, AppearanceEmbedConfig
    from boxing_project.tracking.tracking_stages import (
        PipelineContext, RichStageProgress, PreprocessingStage, LocalTrackingStage,
        LocalDetSavingStage, GlobalClusteringStage, GlobalSavingStage,
    )
    app_embedder = AppearanceEmbedder(AppearanceEmbedConfig(model_path=app_emb_path))
    save_dir = Path(save_dir) if save_dir is not None else Path('output')
    runtime_cfg = pipeline_cfg or {}
    cfg = {
        'stages': runtime_cfg.get('stages', {'preprocessing': True, 'local_tracking': True, 'local_det_saving': False, 'global_clustering': True, 'global_saving': True}),
        'restore_mode': bool(runtime_cfg.get('restore_mode', False)),
        'save_log': bool(runtime_cfg.get('save_log', False)),
        'preprocessing': runtime_cfg.get('preprocessing', {}),
        'local_tracking': runtime_cfg.get('local_tracking', {}),
        'local_det_saving': runtime_cfg.get('local_det_saving', {}),
        'global_clustering': runtime_cfg.get('global_clustering', {}),
        'global_saving': runtime_cfg.get('global_saving', {}),
        'progress': runtime_cfg.get('progress', {'enabled': True, 'library': 'rich'}),
    }
    ctx = PipelineContext(opWrapper=opWrapper, tracker=tracker, app_embedder=app_embedder, sb_cfg=sb_cfg, images=list(images), save_width=save_width, save_dir=save_dir, save_log=cfg['save_log'], restore_mode=cfg['restore_mode'], cfg=cfg, graph_clustering_params=graph_clustering_params or {}, select_top_with_nearest=select_top_with_nearest, extract_features_with_hsv=extract_features_with_hsv, build_fused_appearance_embedding_with_mask=build_fused_appearance_embedding_with_mask)
    progress = RichStageProgress(enabled=bool(cfg.get("progress", {}).get("enabled", True)))
    run_local_det_saving = bool(cfg['stages'].get('local_det_saving', False)) and bool(cfg.get('local_det_saving', {}).get('enabled', True))
    run_global_clustering = bool(cfg['stages'].get('global_clustering', True)) and bool(cfg.get('global_clustering', {}).get('enabled', True))
    run_global_saving = run_global_clustering and bool(cfg['stages'].get('global_saving', True)) and bool(cfg.get('global_saving', {}).get('enabled', True))

    PreprocessingStage(ctx, progress).run()
    LocalTrackingStage(ctx, progress).run()
    if run_local_det_saving:
        LocalDetSavingStage(ctx, progress).run()
    if run_global_clustering:
        GlobalClusteringStage(ctx, progress).run()
        if run_global_saving:
            GlobalSavingStage(ctx, progress).run()
    progress.finish()


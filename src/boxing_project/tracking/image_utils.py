import cv2
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from .track import Detection

# =========================
# OpenPose BODY_25 indices
# =========================

NOSE = 0
NECK = 1
R_SH = 2
L_SH = 5

R_ELBOW = 3
R_WRIST = 4
L_ELBOW = 6
L_WRIST = 7

MID_HIP = 8
R_HIP = 9
R_KNEE = 10
L_HIP = 12
L_KNEE = 13


def _finalize_bbox(x1, y1, x2, y2, img_w, img_h):
    """
    Clip + validate bbox.
    """
    if img_w is not None:
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))

    if img_h is not None:
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def keypoints_to_intersection_bbox(
    kps,
    conf_th: float = 0.1,
    img_w=None,
    img_h=None,
):
    """
    Build a simple full-body bbox from all valid keypoints.

    Used only for overlap / IoU logic.
    Does not return quality.
    """
    if kps is None or len(kps) == 0:
        return None

    kps = np.asarray(kps, dtype=np.float32)

    if kps.ndim != 2 or kps.shape[1] < 2:
        return None

    valid_mask = np.isfinite(kps[:, :2]).all(axis=1)

    if kps.shape[1] >= 3:
        valid_mask &= kps[:, 2] >= float(conf_th)

    if not valid_mask.any():
        return None

    xs = kps[valid_mask, 0]
    ys = kps[valid_mask, 1]

    return _finalize_bbox(
        xs.min(),
        ys.min(),
        xs.max(),
        ys.max(),
        img_w,
        img_h,
    )

def _rects_intersect(r1, r2) -> bool:
    x1, y1, x2, y2 = r1
    X1, Y1, X2, Y2 = r2

    if x2 <= X1 or X2 <= x1:
        return False

    if y2 <= Y1 or Y2 <= y1:
        return False

    return True


BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def bbox_iou(box_a: BBox, box_b: BBox, eps: float = 1e-9) -> float:
    """
    Compute IoU between two bounding boxes.

    Box format:
        (x1, y1, x2, y2)

    Returns:
        IoU in [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= eps:
        return 0.0

    return float(inter_area / union_area)


def get_detection_bbox(det: Detection) -> Optional[BBox]:
    """
    Extract bbox used for bbox debug and overlap scale logic.

    Priority:
        1. det.meta["raw"]["bbox_for_intersection"]
        2. det.meta["raw"]["bbox"]

    Returns:
        bbox as (x1, y1, x2, y2), or None if not available.
    """
    if not isinstance(det.meta, dict):
        return None

    raw = det.meta.get("raw", {})
    if not isinstance(raw, dict):
        return None

    bbox = raw.get("bbox_for_intersection", raw.get("bbox", None))
    return _sanitize_bbox(bbox)


# BODY_25 skeleton edges used for capsule-mask overlap.
# These names are intentionally local to overlap logic so the legacy debug/log
# field names can remain unchanged while their internal meaning changes.
R_ANKLE = 11
L_ANKLE = 14
R_EYE = 15
L_EYE = 16
R_EAR = 17
L_EAR = 18
L_BIG_TOE = 19
L_SMALL_TOE = 20
L_HEEL = 21
R_BIG_TOE = 22
R_SMALL_TOE = 23
R_HEEL = 24

FULL_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (NOSE, NECK),
    (NOSE, R_EYE),
    (NOSE, L_EYE),
    (R_EYE, R_EAR),
    (L_EYE, L_EAR),
    (NECK, R_SH),
    (R_SH, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (NECK, L_SH),
    (L_SH, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (NECK, MID_HIP),
    (MID_HIP, R_HIP),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (MID_HIP, L_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (L_ANKLE, L_BIG_TOE),
    (L_ANKLE, L_SMALL_TOE),
    (L_ANKLE, L_HEEL),
    (R_ANKLE, R_BIG_TOE),
    (R_ANKLE, R_SMALL_TOE),
    (R_ANKLE, R_HEEL),
)

CORE_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (NECK, R_SH),
    (NECK, L_SH),
    (R_SH, L_SH),
    (NECK, MID_HIP),
    (MID_HIP, R_HIP),
    (MID_HIP, L_HIP),
    (R_HIP, L_HIP),
    (R_SH, R_HIP),
    (L_SH, L_HIP),
)


def _coerce_keypoints_and_conf(kps, kp_conf=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return BODY_25-ish xy/conf arrays without raising on malformed input."""
    if kps is None:
        return None, None

    try:
        arr = np.asarray(kps, dtype=np.float32)
    except (TypeError, ValueError):
        return None, None

    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return None, None

    xy = arr[:, :2]

    if kp_conf is not None:
        try:
            conf = np.asarray(kp_conf, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            conf = np.ones((xy.shape[0],), dtype=np.float32)
    elif arr.shape[1] >= 3:
        conf = arr[:, 2].astype(np.float32, copy=False).reshape(-1)
    else:
        conf = np.ones((xy.shape[0],), dtype=np.float32)

    if conf.shape[0] < xy.shape[0]:
        padded = np.zeros((xy.shape[0],), dtype=np.float32)
        padded[: conf.shape[0]] = conf
        conf = padded
    elif conf.shape[0] > xy.shape[0]:
        conf = conf[: xy.shape[0]]

    return xy, conf


def _bbox_from_keypoints(kps, kp_conf=None, conf_threshold: float = 0.05) -> Optional[BBox]:
    xy, conf = _coerce_keypoints_and_conf(kps, kp_conf)
    if xy is None or conf is None:
        return None

    valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf) & (conf >= float(conf_threshold))
    if not bool(valid.any()):
        return None

    pts = xy[valid]
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _sanitize_bbox(bbox) -> Optional[BBox]:
    if bbox is None:
        return None
    try:
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return None

    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def make_skeleton_mask(
    kps,
    kp_conf=None,
    *,
    bbox=None,
    mask_shape: Optional[Tuple[int, int]] = None,
    offset: Tuple[float, float] = (0.0, 0.0),
    core_only: bool = False,
    conf_threshold: float = 0.05,
    thickness: int = 7,
) -> np.ndarray:
    """
    Build a binary capsule skeleton mask from BODY_25 keypoints.

    The helper is deliberately defensive: malformed/missing keypoints,
    confidence, or bbox data returns an empty mask instead of raising.
    """
    xy, conf = _coerce_keypoints_and_conf(kps, kp_conf)
    line_thickness = int(max(1, thickness))

    if mask_shape is None:
        safe_box = _sanitize_bbox(bbox) or _bbox_from_keypoints(kps, kp_conf, conf_threshold)
        if safe_box is None:
            return np.zeros((1, 1), dtype=np.uint8)
        x1, y1, x2, y2 = safe_box
        pad = max(1, line_thickness * 3)
        width = max(1, int(np.ceil(x2 - x1 + 2 * pad)))
        height = max(1, int(np.ceil(y2 - y1 + 2 * pad)))
        mask_shape = (height, width)
        offset = (x1 - pad, y1 - pad)

    try:
        height, width = int(mask_shape[0]), int(mask_shape[1])
    except (TypeError, ValueError, IndexError):
        return np.zeros((1, 1), dtype=np.uint8)

    if height <= 0 or width <= 0:
        return np.zeros((1, 1), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    if xy is None or conf is None:
        return mask

    valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf) & (conf >= float(conf_threshold))
    edges = CORE_SKELETON_EDGES if core_only else FULL_SKELETON_EDGES
    ox, oy = map(float, offset)

    for a, b in edges:
        if a >= xy.shape[0] or b >= xy.shape[0] or not (valid[a] and valid[b]):
            continue

        p1 = (int(round(float(xy[a, 0]) - ox)), int(round(float(xy[a, 1]) - oy)))
        p2 = (int(round(float(xy[b, 0]) - ox)), int(round(float(xy[b, 1]) - oy)))
        cv2.line(mask, p1, p2, color=1, thickness=line_thickness, lineType=cv2.LINE_AA)

    return (mask > 0).astype(np.uint8)


def mask_iou(mask_a, mask_b, eps: float = 1e-9) -> float:
    """Compute binary mask IoU safely for missing/malformed masks."""
    if mask_a is None or mask_b is None:
        return 0.0

    try:
        a = np.asarray(mask_a).astype(bool, copy=False)
        b = np.asarray(mask_b).astype(bool, copy=False)
    except (TypeError, ValueError):
        return 0.0

    if a.shape != b.shape or a.size == 0:
        return 0.0

    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= float(eps):
        return 0.0
    return float(inter / union)


def _pair_mask_geometry(
    kps_a,
    kp_conf_a,
    bbox_a,
    kps_b,
    kp_conf_b,
    bbox_b,
    conf_threshold: float,
    thickness: int,
) -> Tuple[Optional[Tuple[int, int]], Tuple[float, float]]:
    boxes = [
        _sanitize_bbox(bbox_a) or _bbox_from_keypoints(kps_a, kp_conf_a, conf_threshold),
        _sanitize_bbox(bbox_b) or _bbox_from_keypoints(kps_b, kp_conf_b, conf_threshold),
    ]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None, (0.0, 0.0)

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None, (0.0, 0.0)

    line_thickness = int(max(1, thickness))
    pad = max(1, line_thickness * 3)
    width = max(1, int(np.ceil(x2 - x1 + 2 * pad)))
    height = max(1, int(np.ceil(y2 - y1 + 2 * pad)))
    return (height, width), (x1 - pad, y1 - pad)


def _empty_skeleton_overlap() -> Dict[str, float]:
    return {
        "overlap_score": 0.0,
        "skeleton_iou": 0.0,
        "core_skeleton_iou": 0.0,
    }


def skeleton_overlap_score(
    det_a: Detection,
    det_b: Detection,
    *,
    bbox_a=None,
    bbox_b=None,
    full_weight: float = 0.35,
    core_weight: float = 0.65,
    conf_threshold: float = 0.05,
    thickness: int = 7,
) -> Dict[str, float]:
    """Compute weighted BODY_25 capsule overlap for two detections."""
    try:
        box_a = _sanitize_bbox(bbox_a) or get_detection_bbox(det_a)
        box_b = _sanitize_bbox(bbox_b) or get_detection_bbox(det_b)
        shape, offset = _pair_mask_geometry(
            det_a.keypoints,
            det_a.kp_conf,
            box_a,
            det_b.keypoints,
            det_b.kp_conf,
            box_b,
            conf_threshold,
            thickness,
        )
        if shape is None:
            return _empty_skeleton_overlap()

        full_a = make_skeleton_mask(
            det_a.keypoints,
            det_a.kp_conf,
            bbox=box_a,
            mask_shape=shape,
            offset=offset,
            core_only=False,
            conf_threshold=conf_threshold,
            thickness=thickness,
        )
        full_b = make_skeleton_mask(
            det_b.keypoints,
            det_b.kp_conf,
            bbox=box_b,
            mask_shape=shape,
            offset=offset,
            core_only=False,
            conf_threshold=conf_threshold,
            thickness=thickness,
        )
        core_a = make_skeleton_mask(
            det_a.keypoints,
            det_a.kp_conf,
            bbox=box_a,
            mask_shape=shape,
            offset=offset,
            core_only=True,
            conf_threshold=conf_threshold,
            thickness=thickness,
        )
        core_b = make_skeleton_mask(
            det_b.keypoints,
            det_b.kp_conf,
            bbox=box_b,
            mask_shape=shape,
            offset=offset,
            core_only=True,
            conf_threshold=conf_threshold,
            thickness=thickness,
        )

        skeleton_iou = mask_iou(full_a, full_b)
        core_skeleton_iou = mask_iou(core_a, core_b)
        overlap_score = (
            float(full_weight) * skeleton_iou
            + float(core_weight) * core_skeleton_iou
        )
        if not np.isfinite(overlap_score):
            return _empty_skeleton_overlap()

        return {
            "overlap_score": float(overlap_score),
            "skeleton_iou": float(skeleton_iou),
            "core_skeleton_iou": float(core_skeleton_iou),
        }
    except (AttributeError, TypeError, ValueError, IndexError):
        return _empty_skeleton_overlap()


def _bbox_pair_scale(box_a: BBox, box_b: BBox, eps: float = 1e-6) -> float:
    def _scale(box: BBox) -> float:
        x1, y1, x2, y2 = map(float, box)
        return float(np.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1)))

    return max((_scale(box_a) + _scale(box_b)) / 2.0, float(eps))


def _center_dist_norm(det_a: Detection, det_b: Detection, box_a: BBox, box_b: BBox) -> float:
    dist = np.linalg.norm(np.asarray(det_a.center, dtype=float) - np.asarray(det_b.center, dtype=float))
    return float(dist / _bbox_pair_scale(box_a, box_b))


def _normalize_overlap_mechanism(overlap_mechanism: str) -> str:
    if str(overlap_mechanism) == "bbox_iou":
        return "bbox_iou"
    return "skeleton_capsule"


def compute_detection_iou_relations(
    detections: List[Detection],
    overlap_threshold: float = 0.15,
    *,
    overlap_mechanism: str = "skeleton_capsule",
    skeleton_overlap_threshold: float = 0.08,
    skeleton_overlap_full_weight: float = 0.35,
    skeleton_overlap_core_weight: float = 0.65,
    skeleton_overlap_conf_threshold: float = 0.05,
    skeleton_overlap_thickness: int = 7,
    skeleton_overlap_relation_debug_mode: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute broad pairwise geometry relations between detections.

    The output keys are kept for legacy logs/debug UI. When overlap_mechanism
    is "skeleton_capsule", legacy IoU field names carry skeleton/capsule
    overlap scores instead of bbox IoU values.
    """
    mechanism = _normalize_overlap_mechanism(overlap_mechanism)
    active_overlap_threshold = (
        float(skeleton_overlap_threshold)
        if mechanism == "skeleton_capsule"
        else float(overlap_threshold)
    )
    if mechanism == "bbox_iou":
        bboxes = [get_detection_bbox(det) for det in detections]
    else:
        bboxes = [
            get_detection_bbox(det) or _bbox_from_keypoints(
                det.keypoints,
                det.kp_conf,
                skeleton_overlap_conf_threshold,
            )
            for det in detections
        ]
    results: List[Dict[str, Any]] = []

    for i, box_i in enumerate(bboxes):
        max_iou = 0.0
        raw_bbox_max_iou = 0.0
        max_iou_det_idx: Optional[int] = None
        min_center_dist_norm = float("inf")
        center_dist_norm_det_idx: Optional[int] = None
        overlaps: List[Dict[str, Any]] = []

        if box_i is not None:
            for j, box_j in enumerate(bboxes):
                if i == j or box_j is None:
                    continue

                bbox_iou_value = bbox_iou(box_i, box_j)
                skel = skeleton_overlap_score(
                    detections[i],
                    detections[j],
                    bbox_a=box_i,
                    bbox_b=box_j,
                    full_weight=skeleton_overlap_full_weight,
                    core_weight=skeleton_overlap_core_weight,
                    conf_threshold=skeleton_overlap_conf_threshold,
                    thickness=skeleton_overlap_thickness,
                )
                final_overlap_value = (
                    float(skel["overlap_score"])
                    if mechanism == "skeleton_capsule"
                    else float(bbox_iou_value)
                )
                cdn = _center_dist_norm(detections[i], detections[j], box_i, box_j)

                if final_overlap_value > max_iou:
                    max_iou = final_overlap_value
                    max_iou_det_idx = j

                if bbox_iou_value > raw_bbox_max_iou:
                    raw_bbox_max_iou = bbox_iou_value

                if cdn < min_center_dist_norm:
                    min_center_dist_norm = cdn
                    center_dist_norm_det_idx = j

                keep_relation = final_overlap_value > 0.0
                if mechanism == "skeleton_capsule" and bool(skeleton_overlap_relation_debug_mode):
                    keep_relation = keep_relation or bbox_iou_value > 0.0

                if keep_relation:
                    overlaps.append(
                        {
                            "det_idx": int(j),
                            "iou": float(final_overlap_value),
                            "bbox_iou": float(bbox_iou_value),
                            "skeleton_iou": float(skel["skeleton_iou"]),
                            "core_skeleton_iou": float(skel["core_skeleton_iou"]),
                            "center_dist_norm": float(cdn),
                        }
                    )

        results.append(
            {
                "det_idx": int(i),
                "bbox": box_i,
                "max_iou": float(max_iou),
                "raw_bbox_max_iou": float(raw_bbox_max_iou),
                "max_iou_det_idx": max_iou_det_idx,
                "min_center_dist_norm": float(min_center_dist_norm),
                "center_dist_norm_det_idx": center_dist_norm_det_idx,
                "overlaps": overlaps,
                "is_overlapping": bool(max_iou >= active_overlap_threshold),
            }
        )

    return results


def attach_overlap_info_to_detections(
    detections: List[Detection],
    overlap_threshold: float = 0.15,
    *,
    overlap_mechanism: str = "skeleton_capsule",
    skeleton_overlap_threshold: float = 0.08,
    skeleton_overlap_full_weight: float = 0.35,
    skeleton_overlap_core_weight: float = 0.65,
    skeleton_overlap_conf_threshold: float = 0.05,
    skeleton_overlap_thickness: int = 7,
    skeleton_overlap_relation_debug_mode: bool = True,
) -> None:
    """
    Compute overlap relations and attach info to det.meta.

    This mutates detections in-place while preserving legacy field names.
    """
    mechanism = _normalize_overlap_mechanism(overlap_mechanism)
    relations = compute_detection_iou_relations(
        detections=detections,
        overlap_threshold=overlap_threshold,
        overlap_mechanism=mechanism,
        skeleton_overlap_threshold=skeleton_overlap_threshold,
        skeleton_overlap_full_weight=skeleton_overlap_full_weight,
        skeleton_overlap_core_weight=skeleton_overlap_core_weight,
        skeleton_overlap_conf_threshold=skeleton_overlap_conf_threshold,
        skeleton_overlap_thickness=skeleton_overlap_thickness,
        skeleton_overlap_relation_debug_mode=skeleton_overlap_relation_debug_mode,
    )

    for det, info in zip(detections, relations):
        det.meta["max_overlap_iou"] = info["max_iou"]
        det.meta["max_overlap_det_idx"] = info["max_iou_det_idx"]
        det.meta["overlap_relations"] = info["overlaps"]
        det.meta["is_overlapping"] = info["is_overlapping"]
        det.meta["min_center_dist_norm"] = info["min_center_dist_norm"]
        det.meta["center_dist_norm_det_idx"] = info["center_dist_norm_det_idx"]
        det.meta["raw_bbox_max_overlap_iou"] = info.get("raw_bbox_max_iou", 0.0)
        det.meta["overlap_mechanism"] = mechanism


def is_valid_keypoint(kps, keypoint_idx: int, conf_threshold: float) -> bool:
    """
    Перевіряє, чи keypoint можна використовувати.

    Умова:
    1. індекс існує в kps
    2. confidence цієї точки >= conf_threshold

    Очікуваний формат:
        kps[idx] = (x, y, conf)
    """
    return (
        keypoint_idx < len(kps)
        and float(kps[keypoint_idx][2]) >= conf_threshold
    )


def get_keypoint_xy(kps, keypoint_idx: int) -> np.ndarray:
    """
    Повертає координати keypoint у вигляді numpy-вектора [x, y].

    Confidence тут не повертаємо, бо для геометрії нам потрібні тільки координати.
    """
    return np.array(
        [float(kps[keypoint_idx][0]), float(kps[keypoint_idx][1])],
        dtype=np.float32,
    )


def crop_from_bbox(
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Вирізає область з картинки по bbox.

    Що робить:
    1. обрізає координати по межах зображення
    2. переводить їх у int
    3. перевіряє, що bbox має нормальний розмір
    4. повертає crop або None
    """
    image_height, image_width = frame_bgr.shape[:2]

    x1 = int(max(0, min(image_width - 1, round(x1))))
    x2 = int(max(0, min(image_width, round(x2))))
    y1 = int(max(0, min(image_height - 1, round(y1))))
    y2 = int(max(0, min(image_height, round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_bgr[y1:y2, x1:x2]


def crop_glove_from_arm(
    frame_bgr: np.ndarray,
    kps,
    elbow_idx: int,
    wrist_idx: int,
    conf_threshold: float = 0.2,
    forward_shift_ratio: float = 0.25,
    glove_size_ratio: float = 1.35,
):
    """
    Будує crop рукавиці по геометрії руки: elbow -> wrist.

    Ідея:
    - лікоть і зап'ястя задають напрямок передпліччя
    - довжина передпліччя дає локальний масштаб
    - центр crop ставимо трохи далі за wrist
    - розмір crop беремо пропорційно довжині передпліччя
    """
    if not (
        is_valid_keypoint(kps, elbow_idx, conf_threshold)
        and is_valid_keypoint(kps, wrist_idx, conf_threshold)
    ):
        return None

    elbow_xy = get_keypoint_xy(kps, elbow_idx)
    wrist_xy = get_keypoint_xy(kps, wrist_idx)

    # Вектор від ліктя до зап'ястя
    forearm_vector = wrist_xy - elbow_xy

    # Довжина передпліччя в пікселях
    forearm_length = float(np.linalg.norm(forearm_vector))
    if forearm_length < 1e-6:
        return None

    # Одиничний вектор напряму руки
    forearm_direction = forearm_vector / forearm_length

    # Центр crop трохи попереду wrist у напрямку руки
    crop_center = wrist_xy + forward_shift_ratio * forearm_length * forearm_direction

    # Розмір квадрата пропорційний довжині передпліччя
    crop_size = glove_size_ratio * forearm_length

    # Перетворюємо center + size у координати bbox
    x1 = crop_center[0] - crop_size / 2.0
    y1 = crop_center[1] - crop_size / 2.0
    x2 = crop_center[0] + crop_size / 2.0
    y2 = crop_center[1] + crop_size / 2.0

    return crop_from_bbox(frame_bgr, x1, y1, x2, y2)


def crop_shorts_from_hips(
    frame_bgr: np.ndarray,
    kps,
    conf_threshold: float = 0.2,
    shorts_width_ratio: float = 1.5,
    top_lift_ratio: float = 0.35,
    knee_height_ratio: float = 0.45,
    fallback_height_ratio: float = 0.9,
):
    """
    Будує crop шортів по тазових точках.

    Ідея:
    - MID_HIP дає центр шортів
    - L_HIP і R_HIP дають ширину таза
    - верх crop ставимо трохи вище mid_hip
    - низ crop тягнемо вниз:
        * краще по колінах
        * якщо колін нема, то по hip_width
    """
    if not (
        is_valid_keypoint(kps, MID_HIP, conf_threshold)
        and is_valid_keypoint(kps, R_HIP, conf_threshold)
        and is_valid_keypoint(kps, L_HIP, conf_threshold)
    ):
        return None

    mid_hip_xy = get_keypoint_xy(kps, MID_HIP)
    right_hip_xy = get_keypoint_xy(kps, R_HIP)
    left_hip_xy = get_keypoint_xy(kps, L_HIP)

    # Ширина таза
    hip_width = float(np.linalg.norm(left_hip_xy - right_hip_xy))
    if hip_width < 1e-6:
        return None

    # Ширина crop для шортів трохи більша за таз
    shorts_width = shorts_width_ratio * hip_width

    # Верхня межа трохи вище таза
    top_y = mid_hip_xy[1] - top_lift_ratio * hip_width

    # Якщо є коліна, краще оцінюємо низ шортів через відстань до колін
    if (
        is_valid_keypoint(kps, R_KNEE, conf_threshold)
        and is_valid_keypoint(kps, L_KNEE, conf_threshold)
    ):
        right_knee_xy = get_keypoint_xy(kps, R_KNEE)
        left_knee_xy = get_keypoint_xy(kps, L_KNEE)

        average_knee_y = 0.5 * (right_knee_xy[1] + left_knee_xy[1])

        # Беремо частину шляху від таза до колін
        bottom_y = mid_hip_xy[1] + knee_height_ratio * (average_knee_y - mid_hip_xy[1])
    else:
        # Якщо колін нема, просто йдемо вниз на пропорцію від ширини таза
        bottom_y = mid_hip_xy[1] + fallback_height_ratio * hip_width

    # Горизонтально центр беремо по mid_hip
    x1 = mid_hip_xy[0] - shorts_width / 2.0
    x2 = mid_hip_xy[0] + shorts_width / 2.0

    # Вертикально беремо вже готові верх і низ
    y1 = top_y
    y2 = bottom_y

    return crop_from_bbox(frame_bgr, x1, y1, x2, y2)


def extract_boxing_crops(
    frame_bgr: np.ndarray,
    kps,
    conf_threshold: float = 0.4,
):
    """
    Витягує 3 crop-и для одного боксера:
    - left_glove
    - right_glove
    - shorts

    Повертає словник, де кожне значення:
    - np.ndarray, якщо crop вдалося побудувати
    - None, якщо ні
    """
    left_glove_crop = crop_glove_from_arm(
        frame_bgr=frame_bgr,
        kps=kps,
        elbow_idx=L_ELBOW,
        wrist_idx=L_WRIST,
        conf_threshold=conf_threshold,
    )

    right_glove_crop = crop_glove_from_arm(
        frame_bgr=frame_bgr,
        kps=kps,
        elbow_idx=R_ELBOW,
        wrist_idx=R_WRIST,
        conf_threshold=conf_threshold,
    )

    shorts_crop = crop_shorts_from_hips(
        frame_bgr=frame_bgr,
        kps=kps,
        conf_threshold=conf_threshold,
    )

    return {
        "left_glove": left_glove_crop,
        "right_glove": right_glove_crop,
        "shorts": shorts_crop,
    }


def _find_label_position(base_x, base_ty,
                         label_width, label_height,
                         img_w,
                         label_rects,
                         step,
                         min_y):
    """
    Підбирає вертикальну позицію для тексту так, щоб
    його прямокутник не перетинався з уже існуючими в label_rects.
    Повертає (x, baseline_y) і ДОПИСУЄ прямокутник у label_rects.
    """
    # щоб не вилізти за праву межу кадру
    base_x = max(0, min(base_x, img_w - label_width - 1))

    # щоб текст не вилазив за верх і був не нижче min_y
    ty = max(min_y, base_ty)

    while True:
        top = ty - label_height
        bottom = ty
        rect = (base_x, top, base_x + label_width, bottom)

        if top < 0:
            # далі піднімати вже нікуди
            break

        conflict = any(_rects_intersect(rect, r) for r in label_rects)
        if not conflict:
            break

        # піднімаємо текст вище
        ty -= step

    # запам'ятовуємо зайняту зону
    label_rects.append((base_x, ty - label_height, base_x + label_width, ty))
    return base_x, ty




def draw_frame_index(frame: np.ndarray, frame_idx: int,
                     margin: int = 10,
                     font_scale: float = 0.8,
                     thickness: int = 2) -> None:
    """
    Draws "Frame: N" in the top-right corner of the image (in-place).
    """
    text = f"Frame: {frame_idx}"

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    h, w = frame.shape[:2]
    x = max(0, w - tw - margin)
    y = max(th + margin, margin + th)  # baseline y

    # Optional: draw a dark rectangle behind text for readability
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(w, x + tw + pad)
    y2 = min(h, y + baseline + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

def clip_bbox_xyxy(bbox, img_w: int, img_h: int):
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




def expand_bbox_xyxy(
    bbox,
    img_w: int,
    img_h: int,
    width_ratio: float = 0.10,
    height_ratio: float = 0.15,
):
    """Expand bbox around center and clip to image bounds."""
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    if bw <= 0.0 or bh <= 0.0:
        return None

    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5

    new_w = bw * (1.0 + float(width_ratio))
    new_h = bh * (1.0 + float(height_ratio))

    nx1 = int(round(cx - new_w * 0.5))
    nx2 = int(round(cx + new_w * 0.5))
    ny1 = int(round(cy - new_h * 0.5))
    ny2 = int(round(cy + new_h * 0.5))

    return clip_bbox_xyxy((nx1, ny1, nx2, ny2), img_w=img_w, img_h=img_h)


def draw_track_label(frame: np.ndarray, *, x_text: int, ty: int, x1: int, y1: int, track_id: int, det_idx: int, label_height: int = 18, label_prefix: str = "ID") -> None:
    cv2.putText(
        frame,
        f"{label_prefix} {track_id}  Det {det_idx}",
        (x_text, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (36, 255, 12),
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        (x_text, ty - label_height // 2),
        (x1, y1),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_matched_tracks(frame: np.ndarray, detections, matches, label_prefix: str = "ID") -> None:
    """Draw matched bboxes + labels in-place."""
    h, w = frame.shape[:2]

    label_rects = []
    label_height = 18
    label_width_est_id = 60
    step = label_height + 4
    min_y = label_height + 2

    for track_id, det_idx in matches:
        if det_idx < 0 or det_idx >= len(detections):
            continue

        raw = detections[det_idx].meta.get("raw", {})
        bb = clip_bbox_xyxy(raw.get("bbox", None), img_w=w, img_h=h)
        if bb is None:
            continue

        x1, y1, x2, y2 = bb
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        label_width_est = int(label_width_est_id * 1.6)
        x_text, ty = _find_label_position(
            base_x=x1,
            base_ty=y1 - 5,
            label_width=label_width_est,
            label_height=label_height,
            img_w=w,
            label_rects=label_rects,
            step=step,
            min_y=min_y,
        )

        draw_track_label(
            frame,
            x_text=x_text,
            ty=ty,
            x1=x1,
            y1=y1,
            track_id=int(track_id),
            det_idx=int(det_idx),
            label_height=label_height,
            label_prefix=label_prefix,
        )


def render_tracking_overlays(frame: np.ndarray, detections, matches, frame_idx: int, use_global_ids: bool = False) -> None:
    """Main high-level drawing entrypoint for tracking inference."""
    label_prefix = "GID" if use_global_ids else "ID"
    draw_matched_tracks(frame=frame, detections=detections, matches=matches, label_prefix=label_prefix)
    draw_frame_index(frame, frame_idx)

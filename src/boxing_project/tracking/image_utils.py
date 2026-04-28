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


def keypoints_to_bbox(
    kps,
    conf_th=0.1,
    img_w=None,
    img_h=None,
    pose_center_indices=(8, 1, 9, 12, 2, 5),
):
    """
    Build tight and stable body bbox from BODY_25 keypoints.

    Center priority:
      8  - MidHip
      1  - Neck
      9  - Right Hip
      12 - Left Hip
      2  - Right Shoulder
      5  - Left Shoulder

    Goal:
      - use body-root joints as center
      - keep bbox narrow
      - avoid capturing another boxer
      - do NOT expand bbox to include all keypoints
    """

    if kps is None or len(kps) == 0:
        return None, "invalid"

    kps = np.asarray(kps, dtype=np.float32)

    if kps.ndim != 2 or kps.shape[1] < 2:
        return None, "invalid"

    n_kps = len(kps)

    # BODY_25 indices
    NOSE = 0
    NECK = 1
    R_SH = 2
    L_SH = 5
    MID_HIP = 8
    R_HIP = 9
    L_HIP = 12

    def valid(idx):
        return (
            0 <= idx < n_kps
            and np.isfinite(kps[idx, :2]).all()
            and (kps[idx, 2] >= conf_th if kps.shape[1] >= 3 else True)
        )

    def get(idx):
        return kps[idx, :2]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    valid_mask = np.isfinite(kps[:, :2]).all(axis=1)
    if kps.shape[1] >= 3:
        valid_mask &= kps[:, 2] >= conf_th

    if not valid_mask.any():
        return None, "invalid"

    valid_pts = kps[valid_mask, :2]

    # ------------------------------------------------------------
    # 1) Stable root-based center
    # ------------------------------------------------------------
    center_candidates = []

    for idx in pose_center_indices:
        if valid(idx):
            center_candidates.append((idx, get(idx)))

    if center_candidates:
        primary_idx, _ = center_candidates[0]

        centers = []
        weights = []

        for idx, pt in center_candidates:
            if idx == primary_idx:
                w = 4.0
            elif idx == MID_HIP:
                w = 3.0
            elif idx == NECK:
                w = 2.5
            elif idx in (R_HIP, L_HIP):
                w = 1.5
            elif idx in (R_SH, L_SH):
                w = 1.0
            else:
                w = 0.5

            centers.append(pt)
            weights.append(w)

        center = np.average(
            np.asarray(centers, dtype=np.float32),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        ).astype(np.float32)

    else:
        # Last fallback only.
        center = valid_pts.mean(axis=0).astype(np.float32)
        primary_idx = None

    # ------------------------------------------------------------
    # 2) Robust body scale
    # ------------------------------------------------------------
    scales = []

    if valid(R_SH) and valid(L_SH):
        scales.append(dist(get(R_SH), get(L_SH)) * 1.15)

    if valid(R_HIP) and valid(L_HIP):
        scales.append(dist(get(R_HIP), get(L_HIP)) * 1.35)

    if valid(NECK) and valid(MID_HIP):
        scales.append(dist(get(NECK), get(MID_HIP)) * 0.75)

    if valid(NECK) and valid(R_HIP):
        scales.append(dist(get(NECK), get(R_HIP)) * 0.60)

    if valid(NECK) and valid(L_HIP):
        scales.append(dist(get(NECK), get(L_HIP)) * 0.60)

    if scales:
        scale = float(np.median(scales))
        quality = "high" if len(center_candidates) >= 2 else "medium"
    else:
        xs = valid_pts[:, 0]
        ys = valid_pts[:, 1]
        scale = max(
            float(xs.max() - xs.min()),
            float(ys.max() - ys.min()),
            20.0,
        )
        quality = "low"

    scale = max(scale, 20.0)

    # ------------------------------------------------------------
    # 3) Tight bbox around root center
    # ------------------------------------------------------------
    if primary_idx == MID_HIP:
        left = center[0] - 0.65 * scale
        right = center[0] + 0.65 * scale
        top = center[1] - 1.55 * scale
        bottom = center[1] + 0.30 * scale

    elif primary_idx == NECK:
        left = center[0] - 0.68 * scale
        right = center[0] + 0.68 * scale
        top = center[1] - 0.55 * scale
        bottom = center[1] + 1.25 * scale

    elif primary_idx in (R_HIP, L_HIP):
        left = center[0] - 0.70 * scale
        right = center[0] + 0.70 * scale
        top = center[1] - 1.50 * scale
        bottom = center[1] + 0.32 * scale

    elif primary_idx in (R_SH, L_SH):
        left = center[0] - 0.72 * scale
        right = center[0] + 0.72 * scale
        top = center[1] - 0.55 * scale
        bottom = center[1] + 1.30 * scale

    else:
        left = center[0] - 0.72 * scale
        right = center[0] + 0.72 * scale
        top = center[1] - 0.95 * scale
        bottom = center[1] + 1.05 * scale

    # ------------------------------------------------------------
    # 4) Minimal expansion only for weak fallback
    # ------------------------------------------------------------
    if quality == "low":
        pad = 0.03 * scale
        left -= pad
        right += pad
        top -= pad
        bottom += pad

    bbox = _finalize_bbox(left, top, right, bottom, img_w, img_h)

    return bbox, (quality if bbox is not None else "invalid")

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
    Extract bbox from Detection.meta["raw"]["bbox"].

    Expected format:
        det.meta["raw"]["bbox"] = (x1, y1, x2, y2)

    Returns:
        bbox as (x1, y1, x2, y2), or None if not available.
    """
    if not isinstance(det.meta, dict):
        return None

    raw = det.meta.get("raw", {})
    if not isinstance(raw, dict):
        return None

    bbox = raw.get("bbox", None)

    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(float, bbox)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def compute_detection_iou_relations(
    detections: List[Detection],
    overlap_threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise IoU relations between all detections.

    For each detection, returns:
        - max IoU with any other detection
        - index of the most overlapping detection
        - list of overlaps above overlap_threshold
    """
    bboxes = [get_detection_bbox(det) for det in detections]
    results: List[Dict[str, Any]] = []

    for i, box_i in enumerate(bboxes):
        max_iou = 0.0
        max_iou_det_idx: Optional[int] = None
        overlaps: List[Dict[str, Any]] = []

        if box_i is not None:
            for j, box_j in enumerate(bboxes):
                if i == j or box_j is None:
                    continue

                iou = bbox_iou(box_i, box_j)

                if iou > max_iou:
                    max_iou = iou
                    max_iou_det_idx = j

                if iou >= overlap_threshold:
                    overlaps.append(
                        {
                            "det_idx": int(j),
                            "iou": float(iou),
                        }
                    )

        results.append(
            {
                "det_idx": int(i),
                "bbox": box_i,
                "max_iou": float(max_iou),
                "max_iou_det_idx": max_iou_det_idx,
                "overlaps": overlaps,
                "is_overlapping": bool(max_iou >= overlap_threshold),
            }
        )

    return results


def attach_overlap_info_to_detections(
    detections: List[Detection],
    overlap_threshold: float = 0.15,
) -> None:
    """
    Compute IoU relations and attach overlap info to det.meta.

    This mutates detections in-place.

    Added fields:
        det.meta["max_overlap_iou"]
        det.meta["max_overlap_det_idx"]
        det.meta["overlap_relations"]
        det.meta["is_overlapping"]
    """
    relations = compute_detection_iou_relations(
        detections=detections,
        overlap_threshold=overlap_threshold,
    )

    for det, info in zip(detections, relations):
        det.meta["max_overlap_iou"] = info["max_iou"]
        det.meta["max_overlap_det_idx"] = info["max_iou_det_idx"]
        det.meta["overlap_relations"] = info["overlaps"]
        det.meta["is_overlapping"] = info["is_overlapping"]


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

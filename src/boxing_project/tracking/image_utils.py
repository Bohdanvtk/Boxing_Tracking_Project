import cv2
import numpy as np


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


import numpy as np


def keypoints_to_bbox(kps, conf_th=0.1, img_w=None, img_h=None):
    """
    Build tight upper-body bbox (for OSNet) from keypoints.

    Priority:
      1) neck + two shoulders   -> high
      2) neck + one shoulder    -> medium
      3) two shoulders          -> medium
      4) neck + head            -> low
      5) fallback over valid kps -> low

    Returns:
        bbox, quality

    bbox:
        (x1, y1, x2, y2) or None

    quality:
        "high", "medium", "low", or "invalid"
    """
    if kps is None or len(kps) == 0:
        return None, "invalid"

    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[1] < 2:
        return None, "invalid"


    def valid(idx):
        return (
            idx < len(kps)
            and np.isfinite(kps[idx, :2]).all()
            and (kps[idx, 2] >= conf_th if kps.shape[1] >= 3 else True)
        )

    def get(idx):
        return kps[idx, :2]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    has_nose = valid(NOSE)
    has_neck = valid(NECK)
    has_rs = valid(R_SH)
    has_ls = valid(L_SH)

    nose = get(NOSE) if has_nose else None
    neck = get(NECK) if has_neck else None
    rs = get(R_SH) if has_rs else None
    ls = get(L_SH) if has_ls else None

    # CASE 1: neck + both shoulders -> HIGH
    if has_neck and has_rs and has_ls:
        d = dist(rs, ls)
        cx = 0.5 * (rs[0] + ls[0])
        cy = 0.5 * (rs[1] + ls[1])

        w = 1.35 * d

        if has_nose:
            top = nose[1] - 0.25 * d
        else:
            top = neck[1] - 0.65 * d

        bottom = cy + 0.85 * d
        left = cx - w / 2
        right = cx + w / 2

        bbox = _finalize_bbox(left, top, right, bottom, img_w, img_h)
        return bbox, ("high" if bbox is not None else "invalid")

    # CASE 2: neck + one shoulder -> MEDIUM
    if has_neck and (has_rs or has_ls):
        sh = rs if has_rs else ls
        hw = dist(neck, sh)

        w = 2.4 * hw
        cx = neck[0]

        if has_nose:
            top = nose[1] - 0.2 * hw
        else:
            top = neck[1] - 0.75 * hw

        bottom = neck[1] + 1.2 * hw
        left = cx - w / 2
        right = cx + w / 2

        bbox = _finalize_bbox(left, top, right, bottom, img_w, img_h)
        return bbox, ("medium" if bbox is not None else "invalid")

    # CASE 3: two shoulders only -> MEDIUM
    if has_rs and has_ls:
        d = dist(rs, ls)
        cx = 0.5 * (rs[0] + ls[0])
        cy = 0.5 * (rs[1] + ls[1])

        w = 1.3 * d

        if has_nose:
            top = nose[1] - 0.25 * d
        else:
            top = cy - 0.8 * d

        bottom = cy + 0.75 * d
        left = cx - w / 2
        right = cx + w / 2

        bbox = _finalize_bbox(left, top, right, bottom, img_w, img_h)
        return bbox, ("medium" if bbox is not None else "invalid")

    # CASE 4: neck + head -> LOW
    if has_neck and has_nose:
        d = max(dist(neck, nose), 8.0)

        cx = neck[0]
        w = 3.0 * d

        top = nose[1] - 0.3 * d
        bottom = neck[1] + 2.0 * d
        left = cx - w / 2
        right = cx + w / 2

        bbox = _finalize_bbox(left, top, right, bottom, img_w, img_h)
        return bbox, ("low" if bbox is not None else "invalid")

    # CASE 5: fallback over all valid keypoints -> LOW
    valid_mask = (
        kps[:, 2] > conf_th
        if kps.shape[1] >= 3
        else np.ones(len(kps), dtype=bool)
    )

    if not valid_mask.any():
        return None, "invalid"

    xs = kps[valid_mask, 0]
    ys = kps[valid_mask, 1]

    bbox = _finalize_bbox(xs.min(), ys.min(), xs.max(), ys.max(), img_w, img_h)
    return bbox, ("low" if bbox is not None else "invalid")


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

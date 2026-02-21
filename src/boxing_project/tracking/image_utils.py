import cv2
import numpy as np

def keypoints_to_bbox(kps, conf_th=0.1):
    """
    Compute bounding box over keypoints with confidence > conf_th.
    Returns None if no valid keypoints found.
    """
    valid = kps[:, 2] > conf_th
    if not valid.any():
        return None

    xs, ys = kps[valid, 0], kps[valid, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())



def _rects_intersect(r1, r2) -> bool:
    x1, y1, x2, y2 = r1
    X1, Y1, X2, Y2 = r2
    if x2 <= X1 or X2 <= x1:
        return False
    if y2 <= Y1 or Y2 <= y1:
        return False
    return True



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

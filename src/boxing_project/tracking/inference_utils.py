import os
import sys
import cv2
import numpy as np
from pathlib import Path

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



def preprocess_image(opWrapper, img_path: Path, save_width: int, return_img=False):
    """
    Read and resize an image, run OpenPose forward pass.
    Returns Datum (and optionally the resized image).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w = img.shape[:2]
    if w > save_width:
        scale = save_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    datum = op.Datum()
    datum.cvInputData = img

    datums = op.VectorDatum()
    datums.append(datum)

    opWrapper.emplaceAndPop(datums)

    if return_img:
        return datums[0], img
    return datums[0]



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




def process_frame(result, tracker, original_img, conf_th, frame_idx: int):
    """
    Convert OpenPose output to tracker input, update tracker and draw results.
    Returns processed_frame, log_dict.
    """
    frame = original_img.copy()
    h, w = frame.shape[:2]


    if result.poseKeypoints is None:
        return frame, {
            "matches": [],
            "cost_matrix": np.zeros((0, 0)),
            "active_tracks": [],
        }

    kps = result.poseKeypoints
    people = [{"keypoints": kps[i]} for i in range(len(kps))]

    log = tracker.update_with_openpose(people)

    # Precompute all bounding boxes
    bboxes = [keypoints_to_bbox(kps[i], conf_th) for i in range(len(kps))]

    # --------- налаштування для всіх текстових підписів (ID + OP) ---------
    label_rects = []          # тут лежать прямокутники всіх підписів
    label_height = 18
    label_width_est_id = 60   # приблизна ширина "ID 123"
    label_width_est_op = 70   # приблизна ширина "OP 12"
    step = label_height + 4   # крок підняття
    min_y = label_height + 2  # щоб текст не вилазив за верх
    # ----------------------------------------------------------------------

    # ---- Draw tracks: bbox + ID (з урахуванням неперекриття тексту) ----
    for track_id, det_idx in log["matches"]:
        bb = bboxes[det_idx]
        if bb is None:
            continue

        x1, y1, x2, y2 = bb

        # обмежимо bbox рамками зображення
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        # сам bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # шукаємо місце для "ID N"
        base_x_id = x1
        base_ty_id = y1 - 5  # базовий baseline над bbox

        x_text_id, ty_id = _find_label_position(
            base_x=base_x_id,
            base_ty=base_ty_id,
            label_width=label_width_est_id,
            label_height=label_height,
            img_w=w,
            label_rects=label_rects,
            step=step,
            min_y=min_y,
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x_text_id, ty_id),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 255, 12),
            2,
            cv2.LINE_AA,
        )

        # лінія від тексту до верхнього лівого кута bbox
        cv2.line(
            frame,
            (x_text_id, ty_id - label_height // 2),
            (x1, y1),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ---- Draw OpenPose det indices: "OP i" теж без перекриття ----
    for det_idx, bb in enumerate(bboxes):
        if bb is None:
            continue
        x1, y1, _, _ = bb

        base_x_op = x1
        base_ty_op = y1 - 25  # хочемо, щоб "OP" був трохи вище, ніж ID

        x_text_op, ty_op = _find_label_position(
            base_x=base_x_op,
            base_ty=base_ty_op,
            label_width=label_width_est_op,
            label_height=label_height,
            img_w=w,
            label_rects=label_rects,
            step=step,
            min_y=min_y,
        )

        cv2.putText(
            frame,
            f"OP {det_idx}",
            (x_text_op, ty_op),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    draw_frame_index(frame, frame_idx)

    return frame, log



def visualize_sequence(opWrapper, tracker, images, save_width, merge_n):
    frames = []
    count = 0

    debug = bool(getattr(tracker, "debug", False))  # або tracker.cfg.debug

    if debug:
        from boxing_project.tracking.tracking_debug import (
            print_pre_tracking_results,
            print_tracking_results,
        )

    for idx, path in enumerate(images):
        frame_idx = idx + 1

        if debug:
            print_pre_tracking_results(frame_idx)

        result, img = preprocess_image(opWrapper, path, save_width, return_img=True)
        frame, log = process_frame(result, tracker, img, tracker.cfg.min_kp_conf, frame_idx)



        if debug:
            print_tracking_results(log, frame_idx)

        frames.append(frame)
        count += 1

        if count == merge_n and debug:
            _show_merged(frames, merge_n)
            frames = []
            count = 0


def _show_merged(frames, n):
    """
    Merge multiple frames horizontally (aligning by height) and show via cv2.imshow.
    """
    max_h = max(f.shape[0] for f in frames)

    aligned = []
    for f in frames:
        h, w = f.shape[:2]
        if h < max_h:
            top = (max_h - h) // 2
            bottom = max_h - h - top
            f = cv2.copyMakeBorder(f, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        aligned.append(f)

    combined = cv2.hconcat(aligned)

    cv2.imshow(f"Tracking ({n} frames)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from boxing_project.shot_boundary.inference import ShotBoundaryInferConfig, ShotBoundaryInferencer
from boxing_project.tracking.image_utils import _find_label_position, draw_frame_index
from boxing_project.tracking.track import Detection
from boxing_project.tracking.yolo_detector import YoloOnnxPersonDetector


def init_yolo_from_config(yolo_cfg: dict) -> YoloOnnxPersonDetector:
    return YoloOnnxPersonDetector(
        model_path=yolo_cfg["model_path"],
        img_size=int(yolo_cfg.get("img_size", 640)),
        conf_thres=float(yolo_cfg.get("conf_thres", 0.35)),
        iou_thres=float(yolo_cfg.get("iou_thres", 0.5)),
    )


def preprocess_image(img_path: Path, save_width: int, return_img=False):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w = img.shape[:2]
    if w > save_width:
        scale = save_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if return_img:
        return img, img.copy()
    return img


def _save_matched_det(
    *,
    save_dir: Path,
    frame_idx: int,
    track_id: int,
    frame: np.ndarray,
    processed_frame: np.ndarray,
    bbox,
    save_log: bool,
) -> None:
    if save_dir is None:
        return

    h, w = frame.shape[:2]
    frame_dir = Path(save_dir) / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    vis_path = frame_dir / "frame_vis.jpg"
    if processed_frame is not None and not vis_path.exists():
        cv2.imwrite(str(vis_path), processed_frame)

    track_dir = frame_dir / f"track_{track_id}"
    track_dir.mkdir(parents=True, exist_ok=True)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(str(track_dir / "crop.jpg"), crop)

    if save_log:
        from boxing_project.tracking.tracking_debug import GENERAL_LOG

        log_path = save_dir / "debug_log.txt"
        log_path.write_text("\n".join(GENERAL_LOG), encoding="utf-8")


def process_frame(
    frame,
    tracker,
    original_img,
    app_embedder,
    detector: YoloOnnxPersonDetector,
    g: int,
    frame_idx: int,
    reset_mode: bool,
    save_dir: Path | None,
    save_log: bool,
):
    frame = original_img.copy()
    h, w = frame.shape[:2]

    yolo_dets = detector.detect(image_bgr=original_img)
    detections: list[Detection] = []
    for yd in yolo_dets:
        x1, y1, x2, y2 = yd.bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append(
            Detection(
                center=(float(cx), float(cy)),
                bbox_xyxy=yd.bbox_xyxy,
                score=yd.score,
                class_id=yd.class_id,
                meta={"raw": {"bbox": yd.bbox_xyxy, "score": yd.score, "class_id": yd.class_id}},
            )
        )

    for det in detections:
        bbox = det.bbox_xyxy
        if app_embedder is not None and bbox is not None:
            try:
                det.meta["e_app"] = app_embedder.embed(frame, bbox)
            except Exception as e:
                det.meta["e_app"] = None
                det.meta["e_app_error"] = str(e)

    log = tracker.update(detections, g=g, reset_mode=reset_mode)

    label_rects = []
    label_height = 18
    label_width_est_id = 60
    step = label_height + 4
    min_y = label_height + 2

    for track_id, det_idx in log.get("matches", []):
        if det_idx < 0 or det_idx >= len(detections):
            continue

        bb = detections[det_idx].bbox_xyxy
        x1, y1, x2, y2 = bb

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        base_x = x1
        base_ty = y1 - 5
        label_width_est = int(label_width_est_id * 1.6)
        x_text, ty = _find_label_position(
            base_x=base_x,
            base_ty=base_ty,
            label_width=label_width_est,
            label_height=label_height,
            img_w=w,
            label_rects=label_rects,
            step=step,
            min_y=min_y,
        )

        cv2.putText(
            frame,
            f"ID {track_id}  Det {det_idx}",
            (x_text, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 255, 12),
            2,
            cv2.LINE_AA,
        )

        cv2.line(frame, (x_text, ty - label_height // 2), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)

    draw_frame_index(frame, frame_idx)

    if save_dir is not None:
        for track_id, det_idx in log.get("matches", []):
            if det_idx < 0 or det_idx >= len(detections):
                continue

            det = detections[det_idx]
            _save_matched_det(
                save_dir=save_dir,
                frame_idx=frame_idx,
                track_id=int(track_id),
                frame=original_img,
                processed_frame=frame,
                bbox=det.bbox_xyxy,
                save_log=save_log,
            )

    return frame, log


def visualize_sequence(detector, tracker, app_emb_path, sb_cfg: dict, images, save_width, merge_n, save_dir: Path | None):
    debug = tracker.cfg.debug
    save_log = tracker.cfg.save_log

    show_merge = merge_n > 0
    frames = []
    count = 0

    if debug or save_log:
        from boxing_project.tracking.tracking_debug import print_pre_tracking_results, print_tracking_results

    from boxing_project.apperance_embedding.inference import AppearanceEmbedConfig, AppearanceEmbedder

    app_embedder = AppearanceEmbedder(AppearanceEmbedConfig(model_path=app_emb_path))

    sb_cfg = sb_cfg.get("shot_boundary", sb_cfg)
    sb = ShotBoundaryInferencer(
        ShotBoundaryInferConfig(
            resize_w=sb_cfg["resize"][0],
            resize_h=sb_cfg["resize"][1],
            grid_x=sb_cfg["grid"][0],
            grid_y=sb_cfg["grid"][1],
            ema_alpha=sb_cfg.get("ema_alpha", 0.9),
        )
    )

    frame_idx = 0
    for idx, path in enumerate(images):
        frame_idx = idx + 1
        if debug:
            print_pre_tracking_results(frame_idx)

        _, img = preprocess_image(path, save_width, return_img=True)
        g = float(sb.update(img))
        reset_mode = g < float(tracker.cfg.reset_g_threshold)

        frame, log = process_frame(
            frame=None,
            tracker=tracker,
            original_img=img,
            app_embedder=app_embedder,
            detector=detector,
            g=g,
            frame_idx=frame_idx,
            save_dir=save_dir,
            save_log=save_log,
            reset_mode=reset_mode,
        )

        if debug:
            print_tracking_results(log, frame_idx)

        if show_merge:
            frames.append(frame)
            count += 1
            if count == merge_n:
                _show_merged(frames, merge_n)
                frames = []
                count = 0

    print(f"[INFO] {frame_idx} frames were processed")


def _show_merged(frames, n):
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

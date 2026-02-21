import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from boxing_project.tracking.tracker import openpose_people_to_detections
from boxing_project.shot_boundary.inference import ShotBoundaryInferencer, ShotBoundaryInferConfig
from boxing_project.tracking.image_utils import keypoints_to_bbox, expand_bbox_xyxy, render_tracking_overlays
from boxing_project.tracking.saving_utils import FragmentExporter, save_tracking_outputs
from boxing_project.tracking.matcher import build_global_track_mapping


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





@dataclass
class FrameTrackingResult:
    frame_idx: int
    original_img: np.ndarray
    detections: list
    log: dict


def process_frame(result, tracker, original_img, conf_th, app_embedder, g: int, reset_mode: bool):
    """
    Convert OpenPose output to tracker input and run tracking update.
    Returns detections + log without drawing (global IDs are drawn after full sequence).
    """

    if result.poseKeypoints is None:
        return [], {
            "matches": [],
            "cost_matrix": np.zeros((0, 0)),
            "active_tracks": [],
            "epoch_id": int(getattr(tracker, "_epoch_id", 1)),
        }

    kps = result.poseKeypoints  # (N_people, K, 3)
    n_people = len(kps)
    h, w = original_img.shape[:2]

    people = [{"keypoints": kps[i]} for i in range(n_people)]

    bboxes = []
    for i in range(n_people):
        bb = keypoints_to_bbox(kps[i], conf_th)
        bb = expand_bbox_xyxy(bb, img_w=w, img_h=h, width_ratio=0.10, height_ratio=0.15)
        bboxes.append(bb)

    for i in range(n_people):
        people[i]["bbox"] = bboxes[i]

    detections = openpose_people_to_detections(
        people,
        min_kp_conf=tracker.cfg.min_kp_conf,
    )

    for det in detections:
        raw = det.meta.get("raw", {})
        bbox = raw.get("bbox", None)

        if app_embedder is not None and bbox is not None:
            try:
                det.meta["e_app"] = app_embedder.embed(original_img, bbox)
            except Exception as e:
                det.meta["e_app"] = None
                det.meta["e_app_error"] = str(e)

    log = tracker.update(detections, g=g, reset_mode=reset_mode)
    return detections, log


def visualize_sequence(opWrapper, tracker, app_emb_path, sb_cfg: dict, images, save_width, merge_n,
                    save_dir: Path | None):


    debug = tracker.cfg.debug
    save_log = tracker.cfg.save_log

    show_merge = merge_n > 0
    frame_results: list[FrameTrackingResult] = []

    if debug or save_log:
        from boxing_project.tracking.tracking_debug import (
            print_pre_tracking_results,
            print_tracking_results,
        )

    from boxing_project.apperance_embedding.inference import AppearanceEmbedder, AppearanceEmbedConfig

    app_embedder = AppearanceEmbedder(AppearanceEmbedConfig(model_path=app_emb_path))

    fragment_exporter = None
    if save_dir is not None:
        fragment_exporter = FragmentExporter(save_dir / "fragments", min_hits=tracker.cfg.min_hits)

    sb_cfg = sb_cfg.get("shot_boundary", sb_cfg)

    prev_reset_mode = False

    sb = ShotBoundaryInferencer(
        ShotBoundaryInferConfig(
            resize_w=sb_cfg["resize"][0],
            resize_h=sb_cfg["resize"][1],
            grid_x=sb_cfg["grid"][0],
            grid_y=sb_cfg["grid"][1],
            ema_alpha=sb_cfg.get("ema_alpha", 0.9),
        )
    )


    for idx, path in enumerate(images):
        frame_idx = idx + 1

        if debug:
            print_pre_tracking_results(frame_idx)

        result, img = preprocess_image(opWrapper, path, save_width, return_img=True)
        g = float(sb.update(img))

        reset_mode = (g < float(tracker.cfg.reset_g_threshold))

        if reset_mode and not prev_reset_mode and fragment_exporter is not None:
            fragment_exporter.save_tracks(tracker.get_segment_tracks(), frame_idx=frame_idx)

        detections, log = process_frame(
            result, tracker, img,
            tracker.cfg.min_kp_conf,
            app_embedder,
            g=g,
            reset_mode=reset_mode
        )

        frame_results.append(
            FrameTrackingResult(
                frame_idx=frame_idx,
                original_img=img.copy(),
                detections=detections,
                log=log,
            )
        )

        if debug:
            print_tracking_results(log, frame_idx)


        prev_reset_mode = reset_mode

    local_to_global = build_global_track_mapping(
        epoch_tracks=tracker.get_epoch_tracks(),
        large_cost=float(tracker.cfg.match.large_cost),
        greedy_threshold=float(tracker.cfg.match.greedy_threshold),
    )

    rendered_frames = []
    for result in frame_results:
        frame = result.original_img.copy()

        global_matches = []
        for local_track_id, det_idx in result.log.get("matches", []):
            gid = local_to_global.get((int(result.log.get("epoch_id", 1)), int(local_track_id)), int(local_track_id))
            global_matches.append((int(gid), int(det_idx)))

        render_tracking_overlays(
            frame=frame,
            detections=result.detections,
            matches=global_matches,
            frame_idx=result.frame_idx,
            use_global_ids=True,
        )

        if save_dir is not None:
            out_log = dict(result.log)
            out_log["global_matches"] = global_matches
            save_tracking_outputs(
                save_dir=save_dir,
                frame_idx=result.frame_idx,
                original_frame=result.original_img,
                processed_frame=frame,
                detections=result.detections,
                log=out_log,
                conf_th=tracker.cfg.min_kp_conf,
                tracker=tracker,
            )

        rendered_frames.append(frame)

    if show_merge and rendered_frames:
        for i in range(0, len(rendered_frames), merge_n):
            chunk = rendered_frames[i:i + merge_n]
            if chunk:
                _show_merged(chunk, len(chunk))

    print(f"[INFO] {len(frame_results)} frames were processed")


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

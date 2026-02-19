import os
import sys
import cv2
import numpy as np
from pathlib import Path
from boxing_project.tracking.tracker import openpose_people_to_detections
from boxing_project.shot_boundary.inference import ShotBoundaryInferencer, ShotBoundaryInferConfig
from boxing_project.tracking.image_utils import keypoints_to_bbox, expand_bbox_xyxy, render_tracking_overlays
from boxing_project.tracking.saving_utils import FragmentExporter, save_tracking_outputs


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









def process_frame(result, tracker, original_img, conf_th, app_embedder, g: int, frame_idx: int, reset_mode: bool
                  , save_dir: Path | None):
    """
    Convert OpenPose output to tracker input, compute embeddings, update tracker, draw results.
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

    kps = result.poseKeypoints  # (N_people, K, 3)
    n_people = len(kps)

    # 1) people list (raw for detector->detection)
    people = [{"keypoints": kps[i]} for i in range(n_people)]

    # 2) bbox list (index == OpenPose person index)
    # expand slightly for ReID crops: +10% width, +15% height
    bboxes = []
    for i in range(n_people):
        bb = keypoints_to_bbox(kps[i], conf_th)
        bb = expand_bbox_xyxy(bb, img_w=w, img_h=h, width_ratio=0.10, height_ratio=0.15)
        bboxes.append(bb)

    # 3) attach bbox to raw person so it survives into det.meta["raw"]
    for i in range(n_people):
        people[i]["bbox"] = bboxes[i]

    # 4) build detections
    detections = openpose_people_to_detections(
        people,
        min_kp_conf=tracker.cfg.min_kp_conf,
    )

    # 5) compute embeddings and store into det.meta
    #    det.meta already has {"raw": person} from openpose_people_to_detections
    for det in detections:
        raw = det.meta.get("raw", {})
        bbox = raw.get("bbox", None)

        # appearance embedding
        if app_embedder is not None and bbox is not None:
            try:
                det.meta["e_app"] = app_embedder.embed(frame, bbox)
            except Exception as e:
                det.meta["e_app"] = None
                det.meta["e_app_error"] = str(e)

    # 6) update tracker using detections (now they contain embeddings)
    log = tracker.update(detections, g=g, reset_mode=reset_mode)

    render_tracking_overlays(frame=frame, detections=detections, matches=log.get("matches", []), frame_idx=frame_idx)

    if save_dir is not None:
        save_tracking_outputs(
            save_dir=save_dir,
            frame_idx=frame_idx,
            original_frame=original_img,
            processed_frame=frame,
            detections=detections,
            log=log,
            conf_th=conf_th,
            tracker=tracker,
        )

    return frame, log

def visualize_sequence(opWrapper, tracker, app_emb_path, sb_cfg: dict, images, save_width, merge_n,
                    save_dir: Path | None):


    debug = tracker.cfg.debug
    save_log = tracker.cfg.save_log

    show_merge = merge_n > 0
    frames = []
    count = 0

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

        if reset_mode and fragment_exporter is not None:
            fragment_exporter.save_tracks(tracker.get_active_tracks(confirmed_only=False), frame_idx=frame_idx)

        frame, log = process_frame(
            result, tracker, img,
            tracker.cfg.min_kp_conf,
            app_embedder,
            g=g, frame_idx=frame_idx,
            save_dir=save_dir,
            reset_mode=reset_mode
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

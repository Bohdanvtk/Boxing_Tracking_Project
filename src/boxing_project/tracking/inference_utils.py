import os
import sys
import cv2
import numpy as np
from pathlib import Path
from boxing_project.tracking.tracker import openpose_people_to_detections
from boxing_project.shot_boundary.inference import ShotBoundaryInferencer, ShotBoundaryInferConfig
from boxing_project.tracking.image_utils import keypoints_to_bbox, draw_frame_index, _find_label_position


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




def _save_matched_det(
    *,
    save_dir: Path,
    frame_idx: int,
    track_id: int,
    frame: np.ndarray,
    bbox,
    keypoints: np.ndarray | None,
    kp_conf: np.ndarray | None,
    conf_th: float,
) -> None:
    """
    Saves:
      save_dir/frame_000001/track_3/crop.jpg
      save_dir/frame_000001/track_3/kps.npz
    kps saved as (K,4): [x, y, conf, mask]
    """
    if save_dir is None:
        return

    h, w = frame.shape[:2]

    frame_dir = Path(save_dir) / f"frame_{frame_idx:06d}"
    track_dir = frame_dir / f"track_{track_id}"
    track_dir.mkdir(parents=True, exist_ok=True)

    # -------- crop.jpg --------
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(str(track_dir / "crop.jpg"), crop)

    # -------- kps.npz (K,4) --------
    # If no keypoints -> still write empty array for consistency
    if keypoints is None or kp_conf is None:
        kps4 = np.zeros((0, 4), dtype=np.float32)
    else:
        # keypoints expected (K,2) possibly with NaNs; kp_conf (K,)
        xy = keypoints.astype(np.float32, copy=False)
        conf = kp_conf.astype(np.float32, copy=False)

        if xy.ndim != 2 or xy.shape[1] != 2:
            # fail-safe: try reshape if someone passed (K,3) accidentally
            xy = xy.reshape((-1, 2)).astype(np.float32, copy=False)

        conf = conf.reshape((-1,))
        K = min(xy.shape[0], conf.shape[0])
        xy = xy[:K]
        conf = conf[:K]

        # mask: 1 if kp is "present"
        # presence = conf >= conf_th AND coords finite
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        mask = (conf >= float(conf_th)) & finite

        kps4 = np.concatenate(
            [
                xy,
                conf[:, None],
                mask.astype(np.float32)[:, None],
            ],
            axis=1,
        ).astype(np.float32, copy=False)


        #replacing NaN and pos/neg inf with 0.0
        kps4[:, :2] = np.nan_to_num(kps4[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    np.savez_compressed(str(track_dir / "kps.npz"), kps=kps4)

def process_frame(result, tracker, original_img, conf_th, pose_embedder, app_embedder, g: int, frame_idx: int, save_dir: Path | None):
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
    bboxes = [keypoints_to_bbox(kps[i], conf_th) for i in range(n_people)]

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

        # pose embedding
        if pose_embedder is not None and det.keypoints is not None:
            try:
                det.meta["e_pose"] = pose_embedder.embed(det.keypoints, det.kp_conf)
            except Exception as e:
                det.meta["e_pose"] = None
                det.meta["e_pose_error"] = str(e)

        # appearance embedding
        if app_embedder is not None and bbox is not None:
            try:
                det.meta["e_app"] = app_embedder.embed(frame, bbox)
            except Exception as e:
                det.meta["e_app"] = None
                det.meta["e_app_error"] = str(e)

    # 6) update tracker using detections (now they contain embeddings)
    log = tracker.update(detections, g=g)

    # dump matched detections (crop + kps) for this frame
    if save_dir is not None:
        for track_id, det_idx in log.get("matches", []):
            if det_idx < 0 or det_idx >= len(detections):
                continue

            det = detections[det_idx]
            raw = det.meta.get("raw", {})
            bbox = raw.get("bbox", None)

            _save_matched_det(
                save_dir=save_dir,
                frame_idx=frame_idx,
                track_id=int(track_id),
                frame=original_img,  #
                bbox=bbox,
                keypoints=det.keypoints,
                kp_conf=det.kp_conf,
                conf_th=conf_th,
            )

    # --------- label layout settings ---------
    label_rects = []
    label_height = 18
    label_width_est_id = 60
    label_width_est_op = 70
    step = label_height + 4
    min_y = label_height + 2

    # ---- Draw tracks: bbox + ID ----
    for track_id, det_idx in log.get("matches", []):
        # det_idx is index in "detections", NOT necessarily original OpenPose index
        if det_idx < 0 or det_idx >= len(detections):
            continue

        raw = detections[det_idx].meta.get("raw", {})
        bb = raw.get("bbox", None)
        if bb is None:
            continue

        x1, y1, x2, y2 = bb

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        base_x_id = x1
        base_ty_id = y1 - 5

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

        cv2.line(
            frame,
            (x_text_id, ty_id - label_height // 2),
            (x1, y1),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ---- Draw OpenPose det indices: OP i ----
    # NOTE: these are raw OpenPose indices, not detection indices
    for op_idx, bb in enumerate(bboxes):
        if bb is None:
            continue
        x1, y1, _, _ = bb

        base_x_op = int(x1)
        base_ty_op = int(y1) - 25

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
            f"OP {op_idx}",
            (x_text_op, ty_op),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    draw_frame_index(frame, frame_idx)
    return frame, log

def visualize_sequence(opWrapper, tracker, pose_emb_path, app_emb_path, sb_cfg: dict, images, save_width, merge_n,
                    save_dir: Path | None):

    debug = bool(getattr(tracker, "debug", False))  # або tracker.cfg.debug
    show_merge = debug and merge_n > 0
    frames = []
    count = 0

    if debug:
        from boxing_project.tracking.tracking_debug import (
            print_pre_tracking_results,
            print_tracking_results,
        )

    from boxing_project.pose_embeding.inference import PoseEmbedder, PoseEmbedConfig
    from boxing_project.apperance_embedding.inference import AppearanceEmbedder, AppearanceEmbedConfig

    pose_embedder = PoseEmbedder(PoseEmbedConfig(model_path=pose_emb_path))
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


    for idx, path in enumerate(images):
        frame_idx = idx + 1

        if debug:
            print_pre_tracking_results(frame_idx)

        result, img = preprocess_image(opWrapper, path, save_width, return_img=True)
        g = float(sb.update(img))

        frame, log = process_frame(
            result, tracker, img,
            tracker.cfg.min_kp_conf,
            pose_embedder, app_embedder,
            g=g, frame_idx=frame_idx,
            save_dir=save_dir,
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

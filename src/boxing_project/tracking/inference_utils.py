import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from boxing_project.tracking.tracker import openpose_people_to_detections
from boxing_project.shot_boundary.inference import ShotBoundaryInferencer, ShotBoundaryInferConfig
from boxing_project.tracking.image_utils import keypoints_to_bbox, expand_bbox_xyxy, render_tracking_overlays, extract_boxing_crops
from boxing_project.tracking.saving_utils import FragmentExporter, save_tracking_outputs
from boxing_project.tracking.global_clustering import GlobalTrackClusterer
import matplotlib.pyplot as plt



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

def _store_matched_crops(tracker, frame: np.ndarray, detections, matches) -> None:
    tracks_by_id = {int(t.track_id): t for t in tracker.tracks}
    h, w = frame.shape[:2]

    for track_id, det_idx in matches:
        if det_idx < 0 or det_idx >= len(detections):
            continue

        trk = tracks_by_id.get(int(track_id))
        if trk is None:
            continue

        raw = detections[det_idx].meta.get("raw", {})
        bb = _clip_bbox_xyxy(raw.get("bbox", None), img_w=w, img_h=h)
        if bb is None:
            continue

        x1, y1, x2, y2 = bb
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        trk.app_crop_history.append(crop.copy())

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
    Результат обробки одного кадру.
    """
    frame_idx: int
    original_img: np.ndarray
    detections: list
    log: dict

def process_frame(result, tracker, original_img, conf_th, app_embedder, g: int, reset_mode: bool):
    """
    Перетворює OpenPose output у формат, зручний для трекера,
    і паралельно готує crop-и частин тіла для візуальної перевірки.

    Що робить функція:
    1. бере keypoints усіх людей на кадрі
    2. рахує bbox кожної людини
    3. рахує crop-и:
       - left_glove
       - right_glove
       - shorts
    4. записує все у список people
    5. викликає visualize_boxing_crops(...) для дебагу
    """

    # Якщо OpenPose не знайшов жодної людини
    if result.poseKeypoints is None:
        return [], {
            "matches": [],
            "cost_matrix": np.zeros((0, 0)),
            "active_tracks": [],
            "epoch_id": int(getattr(tracker, "_epoch_id", 1)),
        }

    # ------------------------------------------
    # 1. Отримуємо keypoints усіх людей
    # ------------------------------------------
    kps = result.poseKeypoints  # shape: (N_people, K, 3)
    n_people = len(kps)

    # Розмір повного оригінального зображення
    h, w = original_img.shape[:2]

    # Створюємо базову структуру для кожної людини
    people = [{"keypoints": kps[i]} for i in range(n_people)]

    # Тут будемо накопичувати:
    # - bbox кожної людини
    # - crop-и частин тіла
    parts_crops = []
    bboxes = []

    # ------------------------------------------
    # 2. Для кожної людини:
    #    - рахуємо bbox
    #    - рахуємо crop-и
    # ------------------------------------------
    for i in range(n_people):
        # Bbox людини по keypoints
        bb = keypoints_to_bbox(kps[i], conf_th)

        # Crop-и частин тіла:
        # left_glove, right_glove, shorts
        parts = extract_boxing_crops(
            frame_bgr=original_img,
            kps=kps[i],
            conf_threshold=conf_th
        )

        # Трохи розширюємо bbox людини
        bb = expand_bbox_xyxy(
            bb,
            img_w=w,
            img_h=h,
            width_ratio=0.10,
            height_ratio=0.15
        )

        # Зберігаємо результати
        bboxes.append(bb)
        parts_crops.append(parts)

    # ------------------------------------------
    # 3. Записуємо все назад у people
    # ------------------------------------------
    for i in range(n_people):
        people[i]["bbox"] = bboxes[i]

        # Тут тепер уже правильно:
        # це саме crop-и, а не bbox-и
        people[i]["left_glove_crop"] = parts_crops[i]["left_glove"]
        people[i]["right_glove_crop"] = parts_crops[i]["right_glove"]
        people[i]["shorts_crop"] = parts_crops[i]["shorts"]

    for i in range(n_people):
        print(f"Person {i}")

        for part_name, crop in parts_crops[i].items():
            if crop is None:
                print(f"  {part_name}: None")
            else:
                print(f"  {part_name}: {crop.shape}")


    # ------------------------------------------
    # 4. Дебаг-візуалізація
    # ------------------------------------------

    #unactive in this commit
    #visualize_boxing_crops(original_img, people)


    detections = openpose_people_to_detections(
        people,
        min_kp_conf=tracker.cfg.min_kp_conf,
    )

    for det in detections:
        raw = det.meta.get("raw", {})
        bbox = raw.get("bbox", None)
        left_glove_crop = raw.get("left_glove_crop")
        right_glove_crop = raw.get("right_glove_crop")
        shorts_crop = raw.get("shorts_crop")

        if app_embedder is not None and bbox is not None:
            try:
                det.meta["e_app"] = app_embedder.embed(original_img, bbox)
                det.meta["left_glove_features"] = extract_features_with_hsv(left_glove_crop)
                det.meta["right_glove_features"] = extract_features_with_hsv(right_glove_crop)
                det.meta["shorts_features"] = extract_features_with_hsv(shorts_crop)
            except Exception as e:
                det.meta["e_app"] = None
                det.meta["left_glove_features"] = None
                det.meta["right_glove_features"] = None
                det.meta["shorts_features"] = None
                det.meta["e_app_error"] = str(e)

    log = tracker.update(detections, g=g, reset_mode=reset_mode)
    _store_matched_crops(tracker, original_img, detections, log.get("matches", []))
    return detections, log


def visualize_sequence(opWrapper, tracker, app_emb_path, sb_cfg: dict, images, save_width, merge_n,
                    save_dir: Path | None,
                    graph_clustering_params: dict | None = None):


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

    epoch_tracks = tracker.get_epoch_tracks()

    #saving the last fragment(because there is probably no reset_mode in the end)

    if fragment_exporter is not None:
        fragment_exporter.save_tracks(
            tracker.get_segment_tracks(),
            frame_idx=len(images),
        )
    params = graph_clustering_params or {}
    clusterer = GlobalTrackClusterer(
        k=int(params.get("k", 5)),
        sim_threshold=float(params.get("sim_threshold", 0.5)),
        n_clusters=int(params.get("n_clusters", 2)),
        random_state=int(params.get("random_state", 42)),
        assign_labels=str(params.get("assign_labels", "kmeans")),
    )
    local_to_global = clusterer.build_mapping(epoch_tracks=epoch_tracks)

    if save_dir is not None:
        import json

        summary = []
        for epoch_id, tracks_by_id in sorted(epoch_tracks.items()):
            for local_id, trk in sorted(tracks_by_id.items()):
                key = (int(epoch_id), int(local_id))
                gid = local_to_global.get(key)
                summary.append(
                    {
                        "epoch_id": int(epoch_id),
                        "local_track_id": int(local_id),
                        "global_track_id": int(gid) if gid is not None else None,
                        "matched_to_global": bool(gid is not None),
                        "hits": int(getattr(trk, "hits", 0)),
                        "embeddings": int(len(getattr(trk, "app_emb_history", []) or [])),
                        "crops": int(len(getattr(trk, "app_crop_history", []) or [])),
                    }
                )

        (Path(save_dir) / "global_track_mapping.json").write_text(
            json.dumps({"tracks": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rendered_frames = []
    for result in frame_results:
        local_matches = [
            (int(local_track_id), int(det_idx))
            for local_track_id, det_idx in result.log.get("matches", [])
        ]

        local_frame = result.original_img.copy()
        render_tracking_overlays(
            frame=local_frame,
            detections=result.detections,
            matches=local_matches,
            frame_idx=result.frame_idx,
            use_global_ids=False,
        )

        global_frame = result.original_img.copy()
        global_matches = []
        for local_track_id, det_idx in result.log.get("matches", []):
            gid = local_to_global.get((int(result.log.get("epoch_id", 1)), int(local_track_id)), int(local_track_id))
            global_matches.append((int(gid), int(det_idx)))

        render_tracking_overlays(
            frame=global_frame,
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
                processed_frame=global_frame,
                local_processed_frame=local_frame,
                detections=result.detections,
                log=out_log,
                conf_th=tracker.cfg.min_kp_conf,
                tracker=tracker,
            )

        rendered_frames.append(global_frame)

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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import onnxruntime as ort


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class YoloDetection:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_id: int = PERSON_CLASS_ID


class YoloOnnxPersonDetector:
    def __init__(
        self,
        model_path: str | Path,
        img_size: int = 640,
        conf_thres: float = 0.35,
        iou_thres: float = 0.5,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        if boxes.size == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thres]

        return keep

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h0, w0 = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        sx = w0 / self.img_size
        sy = h0 / self.img_size
        return x, sx, sy

    def detect(self, image_bgr: np.ndarray | None = None, image_path: str | Path | None = None) -> List[YoloDetection]:
        if image_bgr is None:
            if image_path is None:
                raise ValueError("Provide image_bgr or image_path")
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

        h0, w0 = image_bgr.shape[:2]
        x, sx, sy = self._preprocess(image_bgr)

        out = self.session.run(None, {self.input_name: x})[0]
        pred = out[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        if cls_scores.size == 0:
            return []

        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        person_mask = (cls_ids == PERSON_CLASS_ID) & (cls_conf >= self.conf_thres)
        if not np.any(person_mask):
            return []

        boxes_xywh = boxes_xywh[person_mask]
        scores = cls_conf[person_mask]

        cx, cy, bw, bh = boxes_xywh.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        keep = self._nms_xyxy(boxes, scores, self.iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]

        detections: List[YoloDetection] = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            ox1 = int(round(float(x1) * sx))
            oy1 = int(round(float(y1) * sy))
            ox2 = int(round(float(x2) * sx))
            oy2 = int(round(float(y2) * sy))

            ox1 = max(0, min(w0 - 1, ox1))
            oy1 = max(0, min(h0 - 1, oy1))
            ox2 = max(0, min(w0 - 1, ox2))
            oy2 = max(0, min(h0 - 1, oy2))
            if ox2 <= ox1 or oy2 <= oy1:
                continue

            detections.append(
                YoloDetection(
                    bbox_xyxy=(ox1, oy1, ox2, oy2),
                    score=float(score),
                )
            )

        return detections

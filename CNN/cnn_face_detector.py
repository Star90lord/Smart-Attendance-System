from pathlib import Path

# Default paths (change as needed)
BASE = Path(__file__).resolve().parent

EMB_DIR = BASE / "embeddings"
DB_PATH = BASE / "attendance.db"

# Detection (choose one):
# 1) YOLOv8 face model weights (.pt). You must provide a face-specific model file path.
#    Example: YOLOv8n-face weights from a reputable source.
YOLO_WEIGHTS = None  # e.g., Path(r"C:/models/yolov8n-face.pt")

# 2) OpenCV DNN SSD fallback (Caffe)
CAFFE_PROTO = None  # e.g., Path(r"C:/models/deploy.prototxt")
CAFFE_MODEL = None  # e.g., Path(r"C:/models/res10_300x300_ssd_iter_140000.caffemodel")

# Thresholds
DET_CONF = 0.40           # detector confidence threshold
SIM_THRESH = 0.55         # cosine similarity threshold for recognition
MARK_COOLDOWN_SEC = 60    # avoid duplicate marks in this window

# Camera
CAM_INDEX = 0
FRAME_W, FRAME_H = 720, 720

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Optional

# YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


class FaceDetector:
    """
    Abstraction for face detection with two backends:
    - YOLOv8 face model (.pt). Use when available.
    - OpenCV DNN SSD (Caffe) fallback (res10_300x300 ...).

    Output boxes are (x1, y1, x2, y2) in image coordinates.
    """

    def __init__(self,
                 yolo_weights: Optional[str] = None,
                 caffe_proto: Optional[str] = None,
                 caffe_model: Optional[str] = None,
                 conf_thresh: float = 0.40):
        self.conf_thresh = float(conf_thresh)
        self.backend = None
        self.yolo = None
        self.net = None

        # Prefer YOLO if a .pt path is provided and ultralytics is installed
        if yolo_weights and _HAS_YOLO:
            self.yolo = YOLO(str(yolo_weights))
            self.backend = "yolo"
        elif caffe_proto and caffe_model:
            self.net = cv2.dnn.readNetFromCaffe(str(caffe_proto), str(caffe_model))
            self.backend = "ssd"
        else:
            raise RuntimeError("No valid detector configured. Provide YOLO .pt weights or Caffe proto+model.")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if self.backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_ssd(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        # Run model on CPU implicitly; use device="cpu" if needed
        results = self.yolo(frame, conf=self.conf_thresh, device="cpu")
        out = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                out.append((int(x1), int(y1), int(x2), int(y2), conf))
        return out

    def _detect_ssd(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.net.setInput(blob)
        dets = self.net.forward()
        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            faces.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2), conf))
        return faces












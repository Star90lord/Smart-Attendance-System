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
FRAME_W, FRAME_H = 1280, 720

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


## recognizer/cnn_recognizer.py
from __future__ import annotations
import os, re, glob
from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1


def ensure_dir(p: str):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def _infer_name_from_path(p: str) -> str:
    base = os.path.basename(p)
    name = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(p))
    if parent and re.search(r"[A-Za-z]", parent):
        return parent
    return re.split(r"[_\-]", name)[0]


def _prep_face(frame_bgr, box, size=160):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5  # [-1,1]
    return rgb.transpose(2, 0, 1)  # CHW


class FaceEmbedder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embed(self, faces_chw: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(faces_chw).to(self.device)
            out = self.model(t).cpu().numpy()
        return out

    def embed_one(self, face_chw: np.ndarray) -> np.ndarray:
        return self.embed(face_chw[None, ...])[0]


def build_embeddings(dataset_dir: str, emb_dir: str, detector, conf_thresh: float = 0.4) -> int:
    """Create one mean-embedding per identity and save as .npy under emb_dir/<name>.npy"""
    ensure_dir(emb_dir)
    embedder = FaceEmbedder(device="cpu")

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    if not imgs:
        print(f"[!] No images found in {dataset_dir}")
        return 0

    per_name_faces: Dict[str, List[np.ndarray]] = {}

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes = detector.detect(img)
        if not boxes:
            continue
        # Use the largest confident box
        boxes = sorted(boxes, key=lambda b: ((b[2]-b[0])*(b[3]-b[1]), b[4]), reverse=True)
        x1, y1, x2, y2, conf = boxes[0]
        if conf < conf_thresh:
            continue
        face = _prep_face(img, (x1, y1, x2, y2))
        if face is None:
            continue
        name = _infer_name_from_path(p)
        per_name_faces.setdefault(name, []).append(face)

    count = 0
    for name, faces in per_name_faces.items():
        arr = np.stack(faces, axis=0)
        embs = embedder.embed(arr)
        mean_emb = embs.mean(axis=0)
        out_path = os.path.join(emb_dir, f"{name}.npy")
        np.save(out_path, mean_emb)
        print(f"[✓] Saved {out_path} from {len(faces)} images")
        count += 1

    print(f"[✓] Built embeddings for {count} identities → {emb_dir}")
    return count


def load_bank(emb_dir: str):
    names, embs = [], []
    for fname in glob.glob(os.path.join(emb_dir, '*.npy')):
        names.append(os.path.splitext(os.path.basename(fname))[0])
        embs.append(np.load(fname))
    if not names:
        print(f"[!] No embeddings found in {emb_dir}")
        return [], np.zeros((0, 512), dtype=np.float32)
    return names, np.stack(embs, axis=0)


def match(emb: np.ndarray, names: list, bank: np.ndarray, sim_thresh: float = 0.55):
    if bank.shape[0] == 0:
        return None, 0.0
    # Cosine similarity
    embn = emb / (np.linalg.norm(emb) + 1e-9)
    bankn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9)
    sims = bankn @ embn
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    return (names[idx], best) if best >= sim_thresh else (None, best)


## db/sql_store.py
import sqlite3
from datetime import datetime
import csv

SCHEMA_STUDENTS = (
    "CREATE TABLE IF NOT EXISTS students ("
    " id INTEGER PRIMARY KEY,"
    " name TEXT UNIQUE NOT NULL,"
    " group_id TEXT,"
    " section TEXT"
    ")"
)

SCHEMA_ATTENDANCE = (
    "CREATE TABLE IF NOT EXISTS attendance ("
    " id INTEGER,"
    " name TEXT,"
    " group_id TEXT,"
    " section TEXT,"
    " ts TEXT,"
    " date TEXT,"
    " status TEXT,"
    " PRIMARY KEY (id, date)"
    ")"
)


def connect(db_path: str):
    return sqlite3.connect(db_path)


def init(conn):
    cur = conn.cursor()
    cur.execute(SCHEMA_STUDENTS)
    cur.execute(SCHEMA_ATTENDANCE)
    conn.commit()


def import_students_csv(conn, csv_path: str):
    cur = conn.cursor()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['id'])
            name = row['name'].strip()
            group_id = row.get('group', row.get('group_id', '')).strip()
            section = row.get('section', '').strip()
            cur.execute(
                "INSERT OR IGNORE INTO students (id, name, group_id, section) VALUES (?, ?, ?, ?)",
                (sid, name, group_id, section)
            )
    conn.commit()


def mark_attendance(conn, sid: int, name: str, group: str, section: str, status: str = "Present"):
    cur = conn.cursor()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    cur.execute(
        "INSERT OR REPLACE INTO attendance (id, name, group_id, section, ts, date, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, name, group, section, ts, date_str, status)
    )
    conn.commit()


def find_student_by_name(conn, name: str):
    cur = conn.cursor()
    cur.execute("SELECT id, name, group_id, section FROM students WHERE name=?", (name,))
    return cur.fetchone()


def export_attendance(conn, for_date: str, out_csv: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id,name,group_id,section,date,status,ts FROM attendance WHERE date=? ORDER BY name", (for_date,))
    rows = cur.fetchall()
    import csv as _csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(["id", "name", "group", "section", "date", "status", "timestamp"])
        w.writerows(rows)
    return len(rows)

## reports/csv_exporter.py
from db.sql_store import export_attendance

def export_for_date(conn, date_str: str, out_csv: str):
    n = export_attendance(conn, date_str, out_csv)
    print(f"[✓] Exported {n} rows → {out_csv}")


## app.py (CLI)

import argparse
import time
import cv2
import numpy as np

from config import (
    EMB_DIR, DB_PATH, YOLO_WEIGHTS, CAFFE_PROTO, CAFFE_MODEL,
    DET_CONF, SIM_THRESH, MARK_COOLDOWN_SEC, CAM_INDEX, FRAME_W, FRAME_H
)

from detector.yolo_detector import FaceDetector
from recognizer.cnn_recognizer import build_embeddings, load_bank, FaceEmbedder, _prep_face, match
from db import sql_store
from reports.csv_exporter import export_for_date


def cmd_register(args):
    # DB init & optional import
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)
    if args.students_csv:
        print(f"[i] Importing students from {args.students_csv}")
        sql_store.import_students_csv(conn, args.students_csv)

    # Detector (YOLO preferred, else SSD)
    det = FaceDetector(yolo_weights=args.yolo_weights or (str(YOLO_WEIGHTS) if YOLO_WEIGHTS else None),
                       caffe_proto=args.proto or (str(CAFFE_PROTO) if CAFFE_PROTO else None),
                       caffe_model=args.caffe_model or (str(CAFFE_MODEL) if CAFFE_MODEL else None),
                       conf_thresh=args.det_conf or DET_CONF)

    # Build embeddings from dataset
    count = build_embeddings(args.dataset_dir, str(EMB_DIR), det, conf_thresh=args.det_conf or DET_CONF)
    print(f"[✓] Registered {count} identities.")
    conn.close()


def cmd_run(args):
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)

    # Detector
    det = FaceDetector(yolo_weights=args.yolo_weights or (str(YOLO_WEIGHTS) if YOLO_WEIGHTS else None),
                       caffe_proto=args.proto or (str(CAFFE_PROTO) if CAFFE_PROTO else None),
                       caffe_model=args.caffe_model or (str(CAFFE_MODEL) if CAFFE_MODEL else None),
                       conf_thresh=args.det_conf or DET_CONF)

    # Bank & embedder
    names, bank = load_bank(str(EMB_DIR))
    embedder = FaceEmbedder(device="cpu")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print("Press 'q' to quit.")
    last_mark = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[!] Camera read failed")
            break

        boxes = det.detect(frame)
        recognized = []

        for (x1, y1, x2, y2, conf) in boxes:
            face = _prep_face(frame, (x1, y1, x2, y2))
            if face is None:
                continue
            emb = embedder.embed_one(face)
            name, sim = match(emb, names, bank, sim_thresh=args.sim_thresh or SIM_THRESH)

            # Draw box + NAME ONLY (as requested)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = name if name else "Unknown"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if name:
                recognized.append(name)

        # Mark attendance with cooldown
        now = time.time()
        for nm in set(recognized):
            if nm not in last_mark or (now - last_mark[nm]) > (args.cooldown or MARK_COOLDOWN_SEC):
                row = sql_store.find_student_by_name(conn, nm)
                if row:
                    sid, name, group_id, section = row
                    sql_store.mark_attendance(conn, sid, name, group_id, section, status="Present")
                    print(f"[✓] Present: {name}")
                else:
                    print(f"[i] Recognized '{nm}' but not found in DB")
                last_mark[nm] = now

        cv2.imshow("Smart Attendance — YOLO + CNN (CPU)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


def cmd_export(args):
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)
    out_csv = args.out_csv or f"attendance_{args.date}.csv"
    export_for_date(conn, args.date, out_csv)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Smart Attendance (YOLOv8 + CNN + SQL + CSV)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # register
    p_reg = sub.add_parser("register", help="Build embeddings from dataset and import students CSV")
    p_reg.add_argument("--dataset_dir", required=True, help="Folder with student images")
    p_reg.add_argument("--students_csv", required=False, help="CSV with id,name,group,section")
    p_reg.add_argument("--yolo_weights", required=False, help="Path to YOLOv8 face .pt weights")
    p_reg.add_argument("--proto", required=False, help="Path to deploy.prototxt (SSD fallback)")
    p_reg.add_argument("--caffe_model", required=False, help="Path to res10_300x300 .caffemodel (SSD fallback)")
    p_reg.add_argument("--det_conf", type=float, required=False)
    p_reg.set_defaults(func=cmd_register)

    # run
    p_run = sub.add_parser("run", help="Live recognition + mark attendance")
    p_run.add_argument("--yolo_weights", required=False, help="Path to YOLOv8 face .pt weights")
    p_run.add_argument("--proto", required=False, help="Path to deploy.prototxt (SSD fallback)")
    p_run.add_argument("--caffe_model", required=False, help="Path to res10_300x300 .caffemodel (SSD fallback)")
    p_run.add_argument("--sim_thresh", type=float, required=False)
    p_run.add_argument("--det_conf", type=float, required=False)
    p_run.add_argument("--cooldown", type=float, required=False)
    p_run.set_defaults(func=cmd_run)

    # export
    p_exp = sub.add_parser("export", help="Export attendance for a date to CSV")
    p_exp.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_exp.add_argument("--out_csv", required=False, help="Output CSV path")
    p_exp.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

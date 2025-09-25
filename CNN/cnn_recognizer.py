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
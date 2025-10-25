#!/usr/bin/env python3
"""
train_model.py
- Loads encodings (either the master encodings_all.pkl OR the per-person .npy files in encodings_by_person/)
- Trains an SVM classifier (sklearn.svm.SVC with probability=True) on all encodings.
- Saves:
    - classifier.pkl
    - label_encoder.pkl (map index -> name using pickle)
    - centroids.npy & centroid_names.pkl (per-person centroid fallback)
Usage:
    python train_model.py --enc-dir ./encodings --out ./model --use-master
    python train_model.py --enc-dir ./encodings --out ./model   # will read per-person .npy files
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_from_master(master_pkl: Path):
    with open(master_pkl, "rb") as f:
        data = pickle.load(f)
    names = list(data["names"])
    encs = [np.array(e) for e in data["encodings"]]
    X = np.stack(encs, axis=0)
    y = np.array(names)
    return X, y

def load_from_perperson(enc_dir: Path):
    per_dir = enc_dir / "encodings_by_person"
    files = sorted([f for f in per_dir.iterdir() if f.suffix == ".npy"])
    X_list = []
    y_list = []
    for f in files:
        name = f.stem
        arr = np.load(f)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        for r in arr:
            X_list.append(r)
            y_list.append(name)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y

def train_and_save(X, y, out_dir: Path, test_size=0.15, random_state=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)

    clf = SVC(kernel="linear", probability=True)
    print("[*] Training SVM...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("[*] Evaluation on held-out set:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classifier.pkl").write_bytes(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL))
    (out_dir / "label_encoder.pkl").write_bytes(pickle.dumps(le, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[*] Saved classifier and label encoder to {out_dir}")

    # Save centroids (mean per person) as a fallback
    centroids = []
    names = []
    for cls in le.classes_:
        mask = (y == cls)
        pts = X[mask]
        cent = np.mean(pts, axis=0)
        centroids.append(cent)
        names.append(cls)
    centroids = np.stack(centroids, axis=0)
    np.save(out_dir / "centroids.npy", centroids)
    (out_dir / "centroid_names.pkl").write_bytes(pickle.dumps(names, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[*] Saved {len(names)} centroids to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-dir", "-e", default="./encodings", help="encodings folder (contains encodings_by_person/ or encodings_all.pkl)")
    parser.add_argument("--out", "-o", default="./model", help="output model folder")
    parser.add_argument("--use-master", action="store_true", help="use encodings_all.pkl (if present) instead of per-person .npy")
    args = parser.parse_args()

    enc_dir = Path(args.enc_dir)
    if args.use_master:
        master = enc_dir / "encodings_all.pkl"
        if not master.exists():
            raise SystemExit(f"{master} not found. Remove --use-master or create master with encode_faces.py")
        X, y = load_from_master(master)
    else:
        per = enc_dir / "encodings_by_person"
        if not per.exists():
            raise SystemExit(f"{per} not found. Run encode_faces.py first.")
        X, y = load_from_perperson(enc_dir)

    print(f"[*] Loaded {X.shape[0]} encodings for {len(np.unique(y))} people.")
    train_and_save(X, y, Path(args.out))

#!/usr/bin/env python3
"""
encode_faces_facenet.py
- Walks dataset_dir where each subfolder is a person (folder name = label).
- For each image, uses facenet-pytorch MTCNN to detect faces, aligns/crops, and InceptionResnetV1 to compute embeddings (512-d).
- Saves per-person encodings -> encodings_by_person/<person>.npy
- Saves master pickle -> encodings_all.pkl: {'names': [...], 'encodings': [...]}
Usage:
    python encode_faces_facenet.py --dataset ./dataset --out ./encodings --allow-multi-face
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def allowed_image(p: Path):
    return p.suffix.lower() in ALLOWED_EXT

def device_for_torch(use_gpu_if_available: bool):
    if use_gpu_if_available and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def encode_dataset(dataset_dir: Path, out_dir: Path, allow_multi_face: bool, device, keep_all_faces: bool):
    mtcnn = MTCNN(keep_all=keep_all_faces, device=device)   # keep_all True returns list of PIL crops per image
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = out_dir / "encodings_by_person"
    per_dir.mkdir(parents=True, exist_ok=True)

    encodings_master = {"names": [], "encodings": []}

    persons = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not persons:
        raise SystemExit(f"No person subfolders found in {dataset_dir}")

    for person in persons:
        name = person.name
        print(f"[+] Processing person: {name}")
        person_encs = []
        images = sorted([f for f in person.iterdir() if f.is_file() and allowed_image(f)])
        if not images:
            print("    - no images, skipping")
            continue

        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"    - failed to open {img_path.name}: {e}")
                continue

            # mtcnn returns a tensor (if keep_all False) or list of tensors (if keep_all True)
            try:
                crops = mtcnn(img)  # if keep_all=False -> Tensor or None; if True -> list or None
            except Exception as e:
                print(f"    - MTCNN error on {img_path.name}: {e}")
                continue

            if crops is None:
                print(f"    - no faces found in {img_path.name}")
                continue

            # normalize to list of tensors
            if isinstance(crops, torch.Tensor):
                crops_list = [crops]
            else:
                crops_list = crops

            if (not allow_multi_face) and len(crops_list) > 1:
                print(f"    - multiple faces ({len(crops_list)}) in {img_path.name}; skipping (use --allow-multi-face to allow)")
                continue

            # compute embeddings in batch on device
            with torch.no_grad():
                batch = torch.stack([c.to(device) for c in crops_list], dim=0)  # shape (N, 3, 160, 160)
                embs = resnet(batch)  # shape (N, 512)
                embs = embs.cpu().numpy()

            for e in embs:
                person_encs.append(np.array(e))

        if person_encs:
            # save per-person npy
            per_file = per_dir / f"{name}.npy"
            np.save(per_file, np.stack(person_encs, axis=0))
            print(f"    - saved {len(person_encs)} encodings to {per_file}")
            for e in person_encs:
                encodings_master['names'].append(name)
                encodings_master['encodings'].append(e)
        else:
            print(f"    - no valid encodings for {name}")

    out_pkl = out_dir / "encodings_all.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(encodings_master, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] Master saved to {out_pkl} ({len(encodings_master['encodings'])} encodings total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, help="dataset root (one folder per person)")
    parser.add_argument("--out", "-o", default="./encodings", help="output folder")
    parser.add_argument("--allow-multi-face", action="store_true", help="allow images with multiple faces")
    parser.add_argument("--use-gpu", action="store_true", help="use GPU if available")
    args = parser.parse_args()

    device = device_for_torch(args.use_gpu)
    encode_dataset(Path(args.dataset), Path(args.out), args.allow_multi_face, device, keep_all_faces=True)

#!/usr/bin/env python3
"""
recognize_webcam_facenet.py
- Webcam-only recognizer using facenet-pytorch (MTCNN + InceptionResnetV1) to create embeddings per-frame
- Loads classifier.pkl and label_encoder.pkl saved by train_model.py
Usage:
    python recognize_webcam_facenet.py --model ./model --enc-dir ./encodings --camera 0 --prob-threshold 0.6 --dist-threshold 0.8 --use-gpu
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

def device_for_torch(use_gpu_if_available: bool):
    if use_gpu_if_available and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class WebcamRecognizer:
    def __init__(self, model_dir: Path, enc_dir: Path, prob_threshold=0.6, dist_threshold=0.8, device=torch.device('cpu')):
        self.classifier = pickle.loads((Path(model_dir)/"classifier.pkl").read_bytes())
        self.label_encoder = pickle.loads((Path(model_dir)/"label_encoder.pkl").read_bytes())
        self.prob_threshold = float(prob_threshold)
        self.dist_threshold = float(dist_threshold)

        # load per-person encodings for distance fallback
        per_dir = Path(enc_dir) / "encodings_by_person"
        self.by_person = {}
        if per_dir.exists():
            for f in per_dir.iterdir():
                if f.suffix == ".npy":
                    name = f.stem
                    arr = np.load(f)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]
                    self.by_person[name] = arr
            print(f"[+] Loaded per-person encodings for {len(self.by_person)} people")
        else:
            print("[!] No per-person encodings found; distance fallback disabled")

        # facenet models
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.device = device

    def min_distance_to_person(self, enc, name):
        arr = self.by_person.get(name)
        if arr is None:
            return float('inf')
        dists = np.linalg.norm(arr - enc, axis=1)
        return float(np.min(dists))

    def predict_encoding(self, enc):
        probs = self.classifier.predict_proba(enc.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        best_name = self.label_encoder.inverse_transform([best_idx])[0]
        best_prob = float(probs[best_idx])
        best_dist = self.min_distance_to_person(enc, best_name)
        if best_prob >= self.prob_threshold and best_dist <= self.dist_threshold:
            return best_name, best_prob, best_dist
        else:
            return "Unknown", best_prob, best_dist

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {camera_index}")
        print("[*] Press 'q' to quit")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # convert to PIL-friendly RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # detect and crop faces (returns list of tensors on device)
                boxes, probs = self.mtcnn.detect(rgb)  # boxes: (N,4) in xyxy
                faces = self.mtcnn.extract(rgb, boxes, save_path=None) if boxes is not None else None

                annotated = frame.copy()
                if faces is not None and len(faces) > 0:
                    # faces are tensors on device; prepare batch
                    with torch.no_grad():
                        if isinstance(faces, list):
                            batch = torch.stack([f.to(self.device) for f in faces], dim=0)
                        else:
                            batch = faces.to(self.device).unsqueeze(0)
                        embs = self.resnet(batch).cpu().numpy()

                    for (box, emb) in zip(boxes, embs):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        name, prob, dist = self.predict_encoding(emb)
                        label = f"{name} ({prob:.2f}, d={dist:.2f})" if name != "Unknown" else f"Unknown (p={prob:.2f}, d={dist:.2f})"
                        color = (0,255,0) if name!="Unknown" else (0,0,255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                cv2.imshow("Webcam Facenet Recognizer", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="./model", help="folder with classifier.pkl and label_encoder.pkl")
    parser.add_argument("--enc-dir", "-e", default="./encodings", help="folder with encodings_by_person")
    parser.add_argument("--camera", "-c", default="0", help="camera index")
    parser.add_argument("--prob-threshold", type=float, default=0.6, help="min classifier probability to accept")
    parser.add_argument("--dist-threshold", type=float, default=0.8, help="max allowed min-distance to person's encodings")
    parser.add_argument("--use-gpu", action="store_true", help="use GPU if available")
    args = parser.parse_args()

    device = device_for_torch(args.use_gpu)
    recognizer = WebcamRecognizer(Path(args.model), Path(args.enc_dir), prob_threshold=args.prob_threshold, dist_threshold=args.dist_threshold, device=device)
    recognizer.run(args.camera)

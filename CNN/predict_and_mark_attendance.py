import argparse
import time
import os
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np

# Keras / TF
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
except Exception as e:
    raise ImportError("TensorFlow/Keras required. Install via `pip install tensorflow`. Error: " + str(e))

# Import detection utilities from facedetect.py (must be colocated)
try:
    from facedetect import (
        load_net,
        detect_multiscale,
        draw_face_detection,
        draw_status_overlay,
        set_cam_resolution,
        save_snapshot,
        save_face_crops,
        DEFAULT_PROTO,
        DEFAULT_MODEL,
    )
except Exception as e:
    raise ImportError(
        "Failed to import utilities from facedetect.py. Place recognision.py in the same folder as facedetect.py.\n"
        "Original error: " + str(e)
    )

# ----------------- Helpers -----------------
def load_class_mapping(mapping_path):
    p = Path(mapping_path)
    if not p.exists():
        raise FileNotFoundError(f"Class mapping JSON not found: {mapping_path}")
    with open(p, "r") as f:
        mapping = json.load(f)
    # mapping: class_name -> index. We need index->class_name
    inv = {int(v): k for k, v in mapping.items()}
    return inv

def predict_face(model, face_img, input_size):
    """Preprocess single face crop and return (label_index, confidence, probs_array)."""
    try:
        img = cv2.resize(face_img, (input_size, input_size))
    except Exception:
        return None, 0.0, None
    arr = img.astype(np.float32)
    arr = mobilenet_preprocess(arr)  # MobileNetV2 preprocessing
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    preds = preds.flatten()
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    return top_idx, top_conf, preds

def append_attendance(csv_path, name, confidence):
    header_needed = not Path(csv_path).exists()
    with open(csv_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["name", "timestamp_iso", "confidence"])
        writer.writerow([name, datetime.now().isoformat(), f"{confidence:.4f}"])

# ----------------- Main recognition loop -----------------
def main():
    parser = argparse.ArgumentParser(description="Real-time face recognition using facedetect and a Keras model.")
    parser.add_argument("--model", default="face_mobilenetv2.h5", help="Path to trained Keras model (.h5)")
    parser.add_argument("--classes", default="face_mobilenetv2.classes.json", help="Path to class mapping JSON")
    parser.add_argument("--proto", default=DEFAULT_PROTO, help="Path to deploy.prototxt for detector")
    parser.add_argument("--caffemodel", default=DEFAULT_MODEL, help="Path to caffemodel for detector")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Face detector min confidence")
    parser.add_argument("--recognition-threshold", type=float, default=0.5, help="Min softmax confidence to accept prediction")
    parser.add_argument("--attendance-out", default="attendance.csv", help="CSV path to append attendance entries")
    parser.add_argument("--attendance-cooldown", type=int, default=60, help="Cooldown seconds before re-logging same person")
    parser.add_argument("--save-crops", default=None, help="Directory to save recognized face crops (optional)")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size expected by model (224 default)")
    args = parser.parse_args()

    # Validate model & mapping
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not Path(args.classes).exists():
        raise FileNotFoundError(f"Class mapping file not found: {args.classes}")

    print("✓ Loading recognition model...")
    model = load_model(args.model)
    print("✓ Loading class mapping...")
    idx2name = load_class_mapping(args.classes)  # index -> name

    print("✓ Loading face detector...")
    net, cuda_used = load_net(args.proto, args.caffemodel, force_cpu=False)
    backend_name = "CUDA" if cuda_used else "CPU"

    # Open camera
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_index}")

    actual_w, actual_h = set_cam_resolution(cap, 1280, 720)
    print(f"✓ Camera resolution: {actual_w}x{actual_h}")
    print(f"✓ Recognition threshold: {args.recognition_threshold:.2f}")
    print(f"✓ Detector min confidence: {args.min_confidence:.2f}")
    print(f"✓ Attendance log: {args.attendance_out}")
    if args.save_crops:
        Path(args.save_crops).mkdir(parents=True, exist_ok=True)
        print(f"✓ Recognized crops will be saved to: {args.save_crops}")
    print("Press 'q' to quit, 's' to take snapshot, 'c' to save current face crops.")
    print("Starting...")

    # Attendance cooldown tracking (name -> datetime last logged)
    last_logged = {}

    frame_count = 0
    fps_history = []

    window_name = "Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            start_ts = time.time()
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            frame_count += 1

            detections = detect_multiscale(net, frame, scales=(1.0, 1.4, 1.8), min_confidence=args.min_confidence)

            # For each detection, predict identity
            displayed = frame.copy()
            recognized_faces = []
            for idx, (conf_det, box) in enumerate(detections, start=1):
                x1, y1, x2, y2 = box
                # expand bounding box slightly for better crop (optional)
                pad = int(0.05 * (x2 - x1))
                xa = max(0, x1 - pad)
                ya = max(0, y1 - pad)
                xb = min(frame.shape[1], x2 + pad)
                yb = min(frame.shape[0], y2 + pad)

                face_crop = frame[ya:yb, xa:xb]
                if face_crop.size == 0:
                    continue

                label_idx, label_conf, _ = predict_face(model, face_crop, args.img_size)
                if label_idx is None:
                    continue

                name = idx2name.get(label_idx, f"class_{label_idx}")

                # Accept only if confidence over threshold
                accepted = label_conf >= args.recognition_threshold

                # Draw detection and label
                display_label = f"{name} {int(label_conf*100)}%" if accepted else f"Unknown {int(label_conf*100)}%"
                # use draw_face_detection if available for consistent look
                try:
                    draw_face_detection(displayed, (xa, ya, xb, yb), label_conf, idx)
                    # put identity label below box
                    cv2.putText(displayed, display_label, (xa, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, lineType=cv2.LINE_AA)
                except Exception:
                    # fallback simple drawing
                    color = (0,255,0) if accepted else (0,180,255)
                    cv2.rectangle(displayed, (xa, ya), (xb, yb), color, 2)
                    cv2.putText(displayed, display_label, (xa, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

                # Registration / attendance logging
                if accepted:
                    # Check cooldown
                    now = datetime.now()
                    last = last_logged.get(name)
                    cooldown_ok = (last is None) or ((now - last).total_seconds() >= args.attendance_cooldown)
                    if cooldown_ok:
                        append_attendance(args.attendance_out, name, label_conf)
                        last_logged[name] = now
                        print(f"✓ Logged attendance: {name} ({label_conf:.3f}) at {now.isoformat()}")

                    # save crop if requested
                    if args.save_crops:
                        ts = int(time.time() * 1000)
                        fname = Path(args.save_crops) / f"{name}_{ts}_{int(label_conf*100)}.jpg"
                        cv2.imwrite(str(fname), face_crop)

                recognized_faces.append((name, label_conf, accepted, (xa, ya, xb, yb)))

            # FPS
            elapsed = time.time() - start_ts
            if elapsed > 0:
                fps = 1.0 / elapsed
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            # overlay status
            try:
                draw_status_overlay(displayed, len(detections), avg_fps, args.min_confidence, backend_name, show_help=False)
            except Exception:
                # fallback small status text
                cv2.putText(displayed, f"Faces: {len(detections)} FPS: {avg_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow(window_name, displayed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("✓ Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                saved = save_snapshot(displayed, output_dir="snapshots")
                if saved:
                    print(f"✓ Snapshot saved: {saved}")
            elif key == ord('c') or key == ord('C'):
                saved_files = save_face_crops(frame, detections, output_dir="face_crops")
                if saved_files:
                    print(f"✓ Saved face crops: {saved_files}")

    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Done.")

if __name__ == "__main__":
    main()
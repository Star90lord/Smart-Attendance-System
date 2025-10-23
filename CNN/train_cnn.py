import os
import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np

# Import facedetect utilities (must be in same folder)
try:
    from facedetect import (
        load_net,
        detect_multiscale,
        set_cam_resolution,
        safe_imwrite,
        DEFAULT_PROTO,
        DEFAULT_MODEL,
    )
except Exception as e:
    raise ImportError(
        "Failed to import from facedetect.py. Place traincnn.py in the same folder as facedetect.py.\n"
        "Original import error: " + str(e)
    )

# TensorFlow / Keras with improved error handling
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    print(f"✓ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    error_msg = str(e)
    print("\n" + "="*70)
    print("❌ TensorFlow Import Error")
    print("="*70)
    print("\nThe error suggests a DLL initialization problem. Try these solutions:\n")
    print("1. Install Microsoft Visual C++ Redistributables:")
    print("   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   - Install and RESTART your computer\n")
    print("2. Reinstall TensorFlow:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow==2.10.0\n")
    print("3. Use Python 3.9 or 3.10 (you're using Python {}.{})".format(
        *map(str, [os.sys.version_info.major, os.sys.version_info.minor])))
    print("\n4. Try CPU-only version:")
    print("   pip install tensorflow-cpu==2.10.0\n")
    print("="*70)
    print(f"\nOriginal error: {error_msg}\n")
    raise SystemExit(1)


# ---------------- Capture samples ----------------
def capture_samples(net, label, output_dir="dataset", samples=250, camera_index=0, img_size=224, min_confidence=0.30, save_augmented=True):
    """
    Capture face crops using detect_multiscale from facedetect.py and save to:
      output_dir/label/*.jpg

    Uses img_size 224 which matches MobileNetV2 input.
    """
    out_path = Path(output_dir) / label
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"✗ Cannot open camera {camera_index}")
        return

    actual_w, actual_h = set_cam_resolution(cap, 1280, 720)
    print(f"✓ Camera opened: {actual_w}x{actual_h}")
    print(f"✓ Capturing ≈{samples} samples for label '{label}' (images will be resized to {img_size}x{img_size})")
    print("Tip: vary angle/pose/lighting. Press 'q' to quit early.")

    count = len(list(out_path.glob("*.jpg")))
    start_time = time.time()

    try:
        while count < samples:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break

            detections = detect_multiscale(net, frame, scales=(1.0, 1.4, 1.8), min_confidence=min_confidence)

            display = frame.copy()
            for idx, (conf, box) in enumerate(detections, start=1):
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{int(conf * 100)}%", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(display, f"Saved: {count}/{samples}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Capture Mode - Press 'q' to stop", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("✗ Capture interrupted by user.")
                break

            if detections:
                # choose highest confidence detection
                detections.sort(key=lambda x: x[0], reverse=True)
                conf, (x1, y1, x2, y2) = detections[0]
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                try:
                    face_resized = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue

                timestamp = int(time.time() * 1000)
                fname = out_path / f"{label}_{timestamp}_{count}.jpg"
                if safe_imwrite(str(fname), face_resized):
                    count += 1

                    if save_augmented:
                        # flip
                        flipped = cv2.flip(face_resized, 1)
                        safe_imwrite(str(out_path / f"{label}_{timestamp}_{count}_flip.jpg"), flipped)
                        # brightness variation
                        factor = 0.85 + np.random.rand() * 0.5
                        bright = np.clip(face_resized.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                        safe_imwrite(str(out_path / f"{label}_{timestamp}_{count}_bright.jpg"), bright)
                        count += 2

                time.sleep(0.08)

    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"✓ Finished capture. Collected {count} samples in {elapsed:.1f}s")
        print(f"✓ Images saved to: {out_path}")


# ---------------- Build & Train (MobileNetV2 transfer) ----------------
def build_mobilenetv2_head(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.4, fine_tune_at=None):
    """
    Build model: MobileNetV2 base (imagenet) + small head.
    fine_tune_at: int layer index in base to start fine-tuning (None = freeze whole base)
    """
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    base.trainable = True  # set True then selectively freeze below
    if fine_tune_at is None:
        # freeze whole base
        for layer in base.layers:
            layer.trainable = False
    else:
        # freeze up to fine_tune_at
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = mobilenet_preprocess(inputs)   # preprocessing consistent with MobileNetV2
    x = base(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


def train_model(data_dir, model_out="face_mobilenetv2.h5", img_size=224, batch_size=32, epochs=15, val_split=0.2, lr=1e-4, fine_tune_at=100):
    """
    Train using MobileNetV2 transfer learning.
    data_dir should have subfolders per-person (labels).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Augmentation and generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.06,
        zoom_range=0.12,
        horizontal_flip=True,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    print(f"✓ Found classes: {train_gen.class_indices}")
    print(f"✓ Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")
    if num_classes < 2:
        print("⚠️ Warning: less than 2 classes found — training may not be meaningful.")

    # Build model with partial fine-tuning
    model = build_mobilenetv2_head(input_shape=(img_size, img_size, 3), num_classes=num_classes, fine_tune_at=fine_tune_at)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_cb = callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint_cb, reduce_lr, early_cb]
    )

    # Save class indices mapping alongside model
    mapping_path = Path(model_out).with_suffix('.classes.json')
    with open(mapping_path, 'w') as f:
        json.dump(train_gen.class_indices, f)
    print(f"✓ Training finished. Best model saved to: {model_out}")
    print(f"✓ Class indices saved to: {mapping_path}")


# ---------------- CLI ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Capture faces and train a MobileNetV2-based face recognizer.")
    sub = parser.add_subparsers(dest="mode", required=True, help="Mode: capture or train")

    cap_p = sub.add_parser("capture", help="Capture face images for a label (uses facedetect)")
    cap_p.add_argument("--label", required=True, help="Label/class name for collected images")
    cap_p.add_argument("--samples", type=int, default=250, help="Target number of samples to collect (approx)")
    cap_p.add_argument("--output", default="dataset", help="Output dataset directory (default ./dataset)")
    cap_p.add_argument("--camera-index", type=int, default=0, help="Camera index")
    cap_p.add_argument("--img-size", type=int, default=224, help="Size to resize face crops (224 for MobileNetV2)")
    cap_p.add_argument("--proto", default=None, help="Path to deploy.prototxt (optional)")
    cap_p.add_argument("--model", default=None, help="Path to caffemodel (optional)")
    cap_p.add_argument("--min-confidence", type=float, default=0.30, help="Detector min confidence")

    train_p = sub.add_parser("train", help="Train MobileNetV2 transfer learning model")
    train_p.add_argument("--data-dir", default=r"C:\Users\PC\Desktop\Smart-Attendance-System\data",
                         help="Dataset directory (default: C:\\Users\\PC\\Desktop\\Smart-Attendance-System\\data)")
    train_p.add_argument("--model-out", default="face_mobilenetv2.h5", help="Output model path (.h5)")
    train_p.add_argument("--img-size", type=int, default=224, help="Image size (224 recommended)")
    train_p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_p.add_argument("--epochs", type=int, default=15, help="Epochs")
    train_p.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (used by ImageDataGenerator)")
    train_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_p.add_argument("--fine-tune-at", type=int, default=100, help="Layer index in base model to start fine-tuning (None to freeze base)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "capture":
        # Use provided proto/model or defaults from facedetect.py
        proto = args.proto if args.proto else DEFAULT_PROTO
        model = args.model if args.model else DEFAULT_MODEL

        net, _cuda = load_net(proto, model, force_cpu=False)
        capture_samples(net, args.label, output_dir=args.output, samples=args.samples,
                        camera_index=args.camera_index, img_size=args.img_size, min_confidence=args.min_confidence)

    elif args.mode == "train":
        train_model(
            data_dir=args.data_dir,
            model_out=args.model_out,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            val_split=args.val_split,
            lr=args.lr,
            fine_tune_at=args.fine_tune_at
        )


if __name__ == "__main__":
    main()
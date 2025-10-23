#!/usr/bin/env python3
"""
Enhanced webcam DNN face detector with:
 - Better small/distant face detection (lower confidence threshold)
 - Clear "Face" label with confidence percentage
 - Multiple detection scales for improved range
 - Image enhancement for better detection
 - Modern UI with corner markers
 - Real-time adjustable confidence

Controls:
    q - quit
    s - save snapshot with detections
    c - save individual face crops
    + - increase confidence (fewer detections)
    - - decrease confidence (more detections, better for distance)
    h - toggle help display
"""

import cv2
import numpy as np
import time
import os
import argparse
from pathlib import Path

# Default model paths
DEFAULT_PROTO = r"C:\Users\PC\Desktop\Smart-Attendance-System\CNN\deploy.prototxt"
DEFAULT_MODEL = r"C:\Users\PC\Desktop\Smart-Attendance-System\CNN\res10_300x300_ssd_iter_140000.caffemodel"

WINDOW_NAME = "Enhanced Face Detector"

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 180, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ------------------ Utility Functions ------------------
def safe_imwrite(path, img):
    """Safely write image to disk."""
    try:
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def set_cam_resolution(cap, w=1280, h=720):
    """Set camera resolution."""
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_w, actual_h
    except Exception:
        return 640, 480

# ------------------ Image Enhancement ------------------
def enhance_image_for_detection(image):
    """
    Enhance image for better face detection.
    Applies CLAHE to improve contrast and handle various lighting.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back to BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

# ------------------ DNN Model Loading ------------------
def load_net(proto_path, model_path, force_cpu=False):
    """Load DNN model with CUDA/CPU backend."""
    if not os.path.isfile(proto_path):
        raise FileNotFoundError(f"Prototxt not found: {proto_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    cuda_used = False
    if not force_cpu:
        try:
            # Try CUDA
            cuda_dev_count = 0
            if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
                cuda_dev_count = cv2.cuda.getCudaEnabledDeviceCount()
            
            if cuda_dev_count > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✓ Using CUDA acceleration")
                cuda_used = True
        except Exception as e:
            print(f"CUDA not available, using CPU: {e}")
    
    if not cuda_used:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✓ Using CPU backend")
    
    return net, cuda_used

def safe_forward(net, blob):
    """Safe forward pass with error handling."""
    net.setInput(blob)
    try:
        return net.forward()
    except cv2.error as e:
        print(f"DNN forward error (trying CPU fallback): {e}")
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net.forward()
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return None

# ------------------ Detection Functions ------------------
def detect_on_image(net, image, min_confidence=0.30):
    """
    Detect faces on a single image/scale.
    Returns list of (confidence, (x1, y1, x2, y2)) tuples.
    """
    h, w = image.shape[:2]
    
    # Create blob for DNN (300x300 input size)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    
    detections = safe_forward(net, blob)
    if detections is None:
        return []
    
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        
        if confidence < min_confidence:
            continue
        
        # Get bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        
        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Filter out tiny boxes
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            results.append((confidence, (x1, y1, x2, y2)))
    
    return results

def detect_multiscale(net, frame, scales=(1.0, 1.4, 1.8, 2.2), min_confidence=0.30):
    """
    Multi-scale detection for better small/distant face detection.
    Detects faces at multiple scales and merges results with NMS.
    """
    h_orig, w_orig = frame.shape[:2]
    
    # Enhance image for better detection
    enhanced_frame = enhance_image_for_detection(frame)
    
    all_boxes = []
    all_scores = []
    
    for scale in scales:
        if scale == 1.0:
            scaled_img = enhanced_frame
        else:
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            scaled_img = cv2.resize(enhanced_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Detect on scaled image
        detections = detect_on_image(net, scaled_img, min_confidence)
        
        # Map coordinates back to original scale
        for conf, (x1, y1, x2, y2) in detections:
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            
            # Clip to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_orig, x2)
            y2 = min(h_orig, y2)
            
            w_box = x2 - x1
            h_box = y2 - y1
            
            if w_box > 10 and h_box > 10:
                all_boxes.append([x1, y1, w_box, h_box])
                all_scores.append(float(conf))
    
    if not all_boxes:
        return []
    
    # Apply Non-Maximum Suppression to remove duplicates
    try:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, min_confidence, 0.3)
        
        if len(indices) == 0:
            return []
        
        # Handle different OpenCV return formats
        if isinstance(indices, tuple):
            indices = indices[0] if len(indices) > 0 else []
        
        indices = np.array(indices).flatten()
        
        chosen = []
        for i in indices:
            conf = all_scores[int(i)]
            x, y, w_box, h_box = all_boxes[int(i)]
            chosen.append((conf, (x, y, x + w_box, y + h_box)))
        
        # Sort by confidence (highest first)
        chosen.sort(key=lambda x: x[0], reverse=True)
        return chosen
    
    except Exception as e:
        print(f"NMS error: {e}")
        # Fallback: return top 10 by confidence
        pairs = sorted(zip(all_scores, all_boxes), key=lambda x: x[0], reverse=True)[:10]
        result = []
        for conf, (x, y, w_box, h_box) in pairs:
            result.append((conf, (x, y, x + w_box, y + h_box)))
        return result

# ------------------ Drawing Functions ------------------
def draw_face_detection(frame, box, confidence, face_id):
    """
    Draw face bounding box with "Face" label and confidence.
    
    Args:
        frame: Image frame
        box: (x1, y1, x2, y2) coordinates
        confidence: Detection confidence
        face_id: Face number ID
    """
    x1, y1, x2, y2 = box
    
    # Choose color based on confidence
    if confidence > 0.7:
        box_color = COLOR_GREEN
    elif confidence > 0.5:
        box_color = COLOR_BLUE
    else:
        box_color = COLOR_YELLOW
    
    # Draw main rectangle
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, lineType=cv2.LINE_AA)
    
    # Draw corner markers for modern look
    corner_length = 20
    corner_thickness = 3
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), box_color, corner_thickness)
    
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), box_color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), box_color, corner_thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), box_color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), box_color, corner_thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), box_color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, corner_thickness)
    
    # Prepare text
    face_label = f"Face #{face_id}"
    conf_label = f"{int(confidence * 100)}%"
    
    # Draw "Face #N" label with background
    draw_text_with_background(frame, face_label, (x1, y1 - 40), 
                             font_scale=0.65, thickness=2, 
                             bg_color=box_color, text_color=COLOR_WHITE)
    
    # Draw confidence percentage below
    draw_text_with_background(frame, conf_label, (x1, y1 - 15),
                             font_scale=0.55, thickness=2,
                             bg_color=COLOR_BLACK, text_color=COLOR_WHITE)

def draw_text_with_background(frame, text, position, font_scale=0.6, thickness=2,
                              bg_color=COLOR_GREEN, text_color=COLOR_WHITE, padding=5):
    """Draw text with solid background rectangle."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(frame,
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding),
                 bg_color, -1, lineType=cv2.LINE_AA)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def draw_status_overlay(frame, num_faces, fps, confidence, backend, show_help=False):
    """Draw status information overlay."""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status text lines
    lines = [
        f"Faces Detected: {num_faces}",
        f"FPS: {fps:.1f} | Backend: {backend}",
        f"Confidence Threshold: {int(confidence * 100)}% (Press +/- to adjust)"
    ]
    
    y_pos = 20
    for line in lines:
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, COLOR_GREEN, 1, lineType=cv2.LINE_AA)
        y_pos += 22
    
    # Draw controls help at bottom if enabled
    if show_help:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        help_lines = [
            "Controls:",
            "Q = Quit | S = Save Snapshot | C = Save Face Crops",
            "+/= Increase Confidence | - Decrease Confidence | H = Toggle Help"
        ]
        
        y_pos = h - 75
        for line in help_lines:
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, COLOR_WHITE, 1, lineType=cv2.LINE_AA)
            y_pos += 22

def save_snapshot(frame, output_dir="snapshots"):
    """Save snapshot with timestamp."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    if safe_imwrite(filepath, frame):
        print(f"✓ Saved snapshot: {filepath}")
        return filepath
    return None

def save_face_crops(frame, detections, output_dir="face_crops"):
    """Save individual face crops."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    saved_files = []
    
    for idx, (conf, (x1, y1, x2, y2)) in enumerate(detections, start=1):
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            continue
        
        filename = f"face_{timestamp}_{idx}_conf{int(conf*100)}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        if safe_imwrite(filepath, face_crop):
            saved_files.append(filepath)
    
    if saved_files:
        print(f"✓ Saved {len(saved_files)} face crops to {output_dir}")
    return saved_files

# ------------------ CLI Arguments ------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Webcam Face Detector")
    parser.add_argument("--proto", default=DEFAULT_PROTO, help="Path to deploy.prototxt")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to caffemodel")
    parser.add_argument("--min-confidence", type=float, default=0.30,
                       help="Minimum confidence (0.2-0.8). Lower = detect distant faces")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.4, 1.8, 2.2],
                       help="Detection scales for distance range")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU backend")
    parser.add_argument("--save-faces", default=None, help="Directory to save face crops")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    return parser.parse_args()

# ------------------ Main Function ------------------
def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 70)
    print("Enhanced Face Detector with Multi-Scale Detection".center(70))
    print("=" * 70)
    
    # Load model
    try:
        net, cuda_used = load_net(args.proto, args.model, args.force_cpu)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Open camera
    try:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera_index)
    except:
        cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print(f"✗ Cannot open camera {args.camera_index}")
        return
    
    # Set resolution
    actual_w, actual_h = set_cam_resolution(cap, 1280, 720)
    print(f"✓ Camera resolution: {actual_w}x{actual_h}")
    print(f"✓ Detection scales: {args.scales}")
    print(f"✓ Min confidence: {args.min_confidence}")
    print("=" * 70)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # State variables
    frame_count = 0
    fps_history = []
    current_confidence = args.min_confidence
    show_help = True
    backend_name = "CUDA" if cuda_used else "CPU"
    
    print("✓ Starting detection... (Press 'H' to toggle help)")
    print("=" * 70)
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect faces with multi-scale detection
            detections = detect_multiscale(net, frame, scales=args.scales, 
                                          min_confidence=current_confidence)
            
            # Draw detections
            for idx, (conf, box) in enumerate(detections, start=1):
                draw_face_detection(frame, box, conf, idx)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = 1.0 / elapsed
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Draw status overlay
            draw_status_overlay(frame, len(detections), avg_fps, 
                              current_confidence, backend_name, show_help)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                save_snapshot(frame)
            elif key == ord('c') or key == ord('C'):
                if args.save_faces:
                    save_face_crops(frame, detections, args.save_faces)
                else:
                    save_face_crops(frame, detections)
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
            elif key == ord('+') or key == ord('='):
                current_confidence = min(0.9, current_confidence + 0.05)
                print(f"Confidence: {current_confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                current_confidence = max(0.1, current_confidence - 0.05)
                print(f"Confidence: {current_confidence:.2f}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("=" * 70)
        print(f"Total frames processed: {frame_count}")
        print("=" * 70)

if __name__ == "__main__":
    main()
# data_cleaning_improved.py
import os
import cv2
import sys
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

# Try to import MTCNN (recommended). If not available, will use OpenCV fallback.
USE_MTCNN = False
try:
    from mtcnn import MTCNN
    USE_MTCNN = True
except Exception:
    USE_MTCNN = False

# Parameters
INPUT_DIR = os.path.join("..", "data")  # keep your original layout
OUTPUT_DIR = "clean_data"
TARGET_SIZE = (224, 224)  # correct target size for your CNN
MIN_FACE_SIZE = 40  # minimum face width in px for detection keep
WORKERS = 8  # adjust for your CPU; set lower if machine is weak

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_face(crop_bgr, out_path):
    try:
        resized = cv2.resize(crop_bgr, TARGET_SIZE)
        cv2.imwrite(out_path, resized)
        return True
    except Exception as e:
        print("Error saving", out_path, e)
        return False

def detect_with_mtcnn(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img_rgb)
    faces = []
    for r in results:
        x, y, w, h = r['box']
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        # ensure positive coords
        x, y = max(0, x), max(0, y)
        faces.append((x, y, w, h))
    return faces

def detect_with_opencv_multi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []

    # Try multiple parameter sets and also detect on resized / equalized images
    trials = [
        {'scaleFactor':1.1, 'minNeighbors':5},
        {'scaleFactor':1.05, 'minNeighbors':3},
        {'scaleFactor':1.2, 'minNeighbors':4},
    ]
    # Also try histogram equalization and a resized version for tiny faces
    gray_eq = cv2.equalizeHist(gray)
    sizes_to_try = [1.0, 0.75, 0.5, 1.25]  # scale image and run detection
    for scale in sizes_to_try:
        if scale != 1.0:
            neww = int(gray_eq.shape[1]*scale)
            newh = int(gray_eq.shape[0]*scale)
            gray_try = cv2.resize(gray_eq, (neww, newh))
        else:
            gray_try = gray_eq
        for params in trials:
            detected = face_cascade.detectMultiScale(gray_try,
                                                     scaleFactor=params['scaleFactor'],
                                                     minNeighbors=params['minNeighbors'],
                                                     minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
            if len(detected) > 0:
                # convert coords back to original scale if we resized
                if scale != 1.0:
                    detected = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x,y,w,h) in detected]
                faces.extend(detected)
        if faces:
            break

    # optionally try flipping (mirror)
    if not faces:
        gray_flip = cv2.flip(gray_eq, 1)
        for params in trials:
            detected = face_cascade.detectMultiScale(gray_flip,
                                                     scaleFactor=params['scaleFactor'],
                                                     minNeighbors=params['minNeighbors'],
                                                     minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
            if len(detected) > 0:
                # flip coords back (x on width)
                w_img = gray_flip.shape[1]
                converted = []
                for (x,y,w,h) in detected:
                    x_original = w_img - x - w
                    converted.append((x_original, y, w, h))
                faces.extend(converted)
            if faces:
                break

    # As last resort try small rotations (±15 degrees)
    if not faces:
        for ang in (-15, 15, -10, 10):
            M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), ang, 1.0)
            rot = cv2.warpAffine(gray_eq, M, (gray.shape[1], gray.shape[0]))
            for params in trials:
                detected = face_cascade.detectMultiScale(rot,
                                                         scaleFactor=params['scaleFactor'],
                                                         minNeighbors=params['minNeighbors'],
                                                         minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                if len(detected) > 0:
                    # invert rotation on boxes: approximate by bounding all rotated boxes back to original
                    faces.extend(detected)
                    break
            if faces:
                break

    # remove duplicates by area/position
    unique = []
    seen = []
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        skip = False
        for s in seen:
            dx = abs(s[0]-center[0]); dy = abs(s[1]-center[1])
            if dx < w*0.5 and dy < h*0.5:
                skip = True; break
        if not skip:
            seen.append(center)
            unique.append((max(0,x), max(0,y), w, h))
    return unique

def process_image(student_output_path, student_name, img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return (img_path, False, "read_error")
        faces = []
        if USE_MTCNN:
            try:
                faces = detect_with_mtcnn(img)
            except Exception as e:
                faces = []
        else:
            faces = detect_with_opencv_multi(img)

        if len(faces) == 0:
            return (img_path, False, "no_face")

        # Save each face; choose largest face first
        faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        saved = 0
        for idx,(x,y,w,h) in enumerate(faces_sorted):
            crop = img[y:y+h, x:x+w]
            base_name = Path(img_path).stem
            out_name = f"{base_name}_face{idx+1}.jpg"
            out_path = os.path.join(student_output_path, out_name)
            ok = save_face(crop, out_path)
            if ok: saved += 1
        return (img_path, True, f"saved_{saved}")
    except Exception as e:
        return (img_path, False, f"error_{e}")

def ensure_folder(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main():
    # validate input folder
    if not os.path.exists(INPUT_DIR):
        print("Input folder not found:", os.path.abspath(INPUT_DIR))
        sys.exit(1)

    failures = []
    tasks = []
    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures = []
        for student_name in os.listdir(INPUT_DIR):
            student_input = os.path.join(INPUT_DIR, student_name)
            student_output = os.path.join(OUTPUT_DIR, student_name)
            if not os.path.isdir(student_input):
                continue
            ensure_folder(student_output)

            for img_name in os.listdir(student_input):
                img_path = os.path.join(student_input, img_name)
                # schedule worker
                futures.append(exe.submit(process_image, student_output, student_name, img_path))

        # collect results
        for fut in as_completed(futures):
            img_path, ok, info = fut.result()
            if not ok:
                failures.append((img_path, info))
            else:
                # optionally print saved info for debugging
                print("OK:", img_path, info)

    # write failures log
    failures_csv = os.path.join(OUTPUT_DIR, "failures.csv")
    with open(failures_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["image", "reason"])
        for img, reason in failures:
            w.writerow([img, reason])
    print("Done. Failures logged to:", failures_csv)
    print("If many 'no_face' failures, consider increasing MIN_FACE_SIZE or installing MTCNN with `pip install mtcnn`")

if __name__ == "__main__":
    print("Using MTCNN?" , USE_MTCNN)
    main()

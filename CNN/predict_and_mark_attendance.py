import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from face_detector import detect_faces  # your DNN face detector

# ----------------- CONFIG -----------------
MODEL_DIR = "models"
MODEL_NAME = "smart_attendance_effb0"
DATA_DIR = r"C:\Users\PC\Desktop\Smart-Attendance-System\data"
LABELS_CSV = os.path.join(DATA_DIR, "details.csv")
IMG_SIZE = (160, 160)
THRESHOLD = 0.80
ATTENDANCE_CSV = "attendance.csv"

# ----------------- LOAD MODEL & LABELS -----------------
model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.keras")
config_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_config.json")
label_table_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_label_table.csv")

model = load_model(model_path)

# Load label table with metadata
label_table = pd.read_csv(label_table_path)
if os.path.exists(LABELS_CSV):
    details_df = pd.read_csv(LABELS_CSV)
    label_table = label_table.merge(details_df, how="left", on="Person_Name")

# Map class_index -> metadata
class_metadata = {}
for idx, row in label_table.iterrows():
    class_metadata[int(row["class_index"])] = {
        "system_id": row.get("System_ID", row.get("system_id", "Unknown")),
        "group": row.get("Group", row.get("group", "Unknown")),
        "section": row.get("Section", "Unknown") if "Section" in row else "Unknown",
        "person_name": row.get("Person_Name", row.get("person_name", "Unknown"))
    }

# ----------------- ATTENDANCE LOG -----------------
attendance_records = {}  # keep track of attendance in this session

# If CSV exists, load existing data to avoid duplicates
if os.path.exists(ATTENDANCE_CSV):
    df_existing = pd.read_csv(ATTENDANCE_CSV)
    for _, row in df_existing.iterrows():
        attendance_records[row["system_id"]] = row.to_dict()

cap = cv2.VideoCapture(0)
print("Press 'q' to quit. Attendance will be marked and saved automatically in real-time.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    boxes, faces = detect_faces(frame)

    for i, face in enumerate(faces):
        x1, y1, x2, y2, conf = boxes[i]

        # Preprocess for CNN
        face_input = cv2.resize(face, IMG_SIZE)
        face_input = face_input.astype("float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        preds = model.predict(face_input, verbose=0)
        class_idx = np.argmax(preds)
        pred_conf = np.max(preds)

        if pred_conf >= THRESHOLD:
            metadata = class_metadata[class_idx]
            sys_id = metadata["system_id"]

            # Only log once per session
            if sys_id not in attendance_records:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_records[sys_id] = {
                    "timestamp": timestamp,
                    "system_id": metadata["system_id"],
                    "group": metadata["group"],
                    "section": metadata["section"],
                    "person_name": metadata["person_name"]
                }
                # Update CSV in real-time
                df = pd.DataFrame(attendance_records.values())
                df.to_csv(ATTENDANCE_CSV, index=False)
                print(f"Marked attendance: {metadata['person_name']} at {timestamp}")

            label_text = f"{metadata['person_name']} ({pred_conf*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Unknown ({pred_conf*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Attendance session ended. Data saved to {ATTENDANCE_CSV}")

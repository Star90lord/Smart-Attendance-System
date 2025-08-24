from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on CPU
    results = model(frame, conf=0.4, device="cpu")

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            w, h = int(x2 - x1), int(y2 - y1)

            # ✅ Expand/shrink box by a factor
            margin_w = int(w * 0.2)   # 20% extra width
            margin_h = int(h * 0.3)   # 30% extra height

            # Adjusted coordinates
            new_x1 = max(0, int(x1 - margin_w))
            new_y1 = max(0, int(y1 - margin_h))
            new_x2 = min(frame.shape[1], int(x2 + margin_w))
            new_y2 = min(frame.shape[0], int(y2 + margin_h))

            # Draw adjusted rectangle
            cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf*100:.1f}%", (new_x1, new_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("YOLOv8 Face Detection (Adjusted)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

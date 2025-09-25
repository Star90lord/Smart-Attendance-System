## app.py (CLI)
import argparse
import time
import cv2
import numpy as np

from config import (
    EMB_DIR, DB_PATH, YOLO_WEIGHTS, CAFFE_PROTO, CAFFE_MODEL,
    DET_CONF, SIM_THRESH, MARK_COOLDOWN_SEC, CAM_INDEX, FRAME_W, FRAME_H
)

from cnn_face_detector import FaceDetector
from cnn_recognizer import build_embeddings, load_bank, FaceEmbedder, _prep_face, match
from sql_store import *
from csv_exporter import export_for_date



def cmd_register(args):
    # DB init & optional import
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)
    if args.students_csv:
        print(f"[i] Importing students from {args.students_csv}")
        sql_store.import_students_csv(conn, args.students_csv)

    # Detector (YOLO preferred, else SSD)
    det = FaceDetector(yolo_weights=args.yolo_weights or (str(YOLO_WEIGHTS) if YOLO_WEIGHTS else None),
                       caffe_proto=args.proto or (str(CAFFE_PROTO) if CAFFE_PROTO else None),
                       caffe_model=args.caffe_model or (str(CAFFE_MODEL) if CAFFE_MODEL else None),
                       conf_thresh=args.det_conf or DET_CONF)

    # Build embeddings from dataset
    count = build_embeddings(args.dataset_dir, str(EMB_DIR), det, conf_thresh=args.det_conf or DET_CONF)
    print(f"[✓] Registered {count} identities.")
    conn.close()


def cmd_run(args):
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)

    # Detector
    det = FaceDetector(yolo_weights=args.yolo_weights or (str(YOLO_WEIGHTS) if YOLO_WEIGHTS else None),
                       caffe_proto=args.proto or (str(CAFFE_PROTO) if CAFFE_PROTO else None),
                       caffe_model=args.caffe_model or (str(CAFFE_MODEL) if CAFFE_MODEL else None),
                       conf_thresh=args.det_conf or DET_CONF)

    # Bank & embedder
    names, bank = load_bank(str(EMB_DIR))
    embedder = FaceEmbedder(device="cpu")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print("Press 'q' to quit.")
    last_mark = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[!] Camera read failed")
            break

        boxes = det.detect(frame)
        recognized = []

        for (x1, y1, x2, y2, conf) in boxes:
            face = _prep_face(frame, (x1, y1, x2, y2))
            if face is None:
                continue
            emb = embedder.embed_one(face)
            name, sim = match(emb, names, bank, sim_thresh=args.sim_thresh or SIM_THRESH)

            # Draw box + NAME ONLY (as requested)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = name if name else "Unknown"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if name:
                recognized.append(name)

        # Mark attendance with cooldown
        now = time.time()
        for nm in set(recognized):
            if nm not in last_mark or (now - last_mark[nm]) > (args.cooldown or MARK_COOLDOWN_SEC):
                row = sql_store.find_student_by_name(conn, nm)
                if row:
                    sid, name, group_id, section = row
                    sql_store.mark_attendance(conn, sid, name, group_id, section, status="Present")
                    print(f"[✓] Present: {name}")
                else:
                    print(f"[i] Recognized '{nm}' but not found in DB")
                last_mark[nm] = now

        cv2.imshow("Smart Attendance — YOLO + CNN (CPU)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


def cmd_export(args):
    conn = sql_store.connect(str(DB_PATH))
    sql_store.init(conn)
    out_csv = args.out_csv or f"attendance_{args.date}.csv"
    export_for_date(conn, args.date, out_csv)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Smart Attendance (YOLOv8 + CNN + SQL + CSV)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # register
    p_reg = sub.add_parser("register", help="Build embeddings from dataset and import students CSV")
    p_reg.add_argument("--dataset_dir", required=True, help="Folder with student images")
    p_reg.add_argument("--students_csv", required=False, help="CSV with id,name,group,section")
    p_reg.add_argument("--yolo_weights", required=False, help="Path to YOLOv8 face .pt weights")
    p_reg.add_argument("--proto", required=False, help="Path to deploy.prototxt (SSD fallback)")
    p_reg.add_argument("--caffe_model", required=False, help="Path to res10_300x300 .caffemodel (SSD fallback)")
    p_reg.add_argument("--det_conf", type=float, required=False)
    p_reg.set_defaults(func=cmd_register)

    # run
    p_run = sub.add_parser("run", help="Live recognition + mark attendance")
    p_run.add_argument("--yolo_weights", required=False, help="Path to YOLOv8 face .pt weights")
    p_run.add_argument("--proto", required=False, help="Path to deploy.prototxt (SSD fallback)")
    p_run.add_argument("--caffe_model", required=False, help="Path to res10_300x300 .caffemodel (SSD fallback)")
    p_run.add_argument("--sim_thresh", type=float, required=False)
    p_run.add_argument("--det_conf", type=float, required=False)
    p_run.add_argument("--cooldown", type=float, required=False)
    p_run.set_defaults(func=cmd_run)

    # export
    p_exp = sub.add_parser("export", help="Export attendance for a date to CSV")
    p_exp.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_exp.add_argument("--out_csv", required=False, help="Output CSV path")
    p_exp.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from typing import List, Tuple, Optional
import sqlite3
from datetime import datetime
import time
import logging
from pydantic import BaseModel
from detector.yolo_detector import FaceDetector
from recognizer.cnn_recognizer import FaceEmbedder, load_bank, match, _prep_face
from db.sql_store import connect, init, mark_attendance, find_student_by_name

# Configuration
BASE = Path(__file__).resolve().parent
EMB_DIR = BASE / "embeddings"
DB_PATH = BASE / "attendance.db"
YOLO_WEIGHTS = None  
CAFFE_PROTO = None   
CAFFE_MODEL = None  
DET_CONF = 0.40
SIM_THRESH = 0.55
MARK_COOLDOWN_SEC = 60
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Face Detection and Recognition Backend")

# Global variables
detector = None
embedder = None
names = []
bank = None
last_mark = {}
db_conn = None

class DetectionResponse(BaseModel):
    faces: List[Tuple[int, int, int, int, float]]
    names: List[Optional[str]]
    confidences: List[float]

def initialize():
    global detector, embedder, names, bank, db_conn
    try:
        # Initialize detector
        detector = FaceDetector(
            yolo_weights=YOLO_WEIGHTS,
            caffe_proto=CAFFE_PROTO,
            caffe_model=CAFFE_MODEL,
            conf_thresh=DET_CONF
        )
        logger.info(f"Initialized detector with backend: {detector.backend}")
        
        # Initialize embedder and load embeddings
        embedder = FaceEmbedder(device="cpu")
        names, bank = load_bank(str(EMB_DIR))
        logger.info(f"Loaded {len(names)} embeddings from {EMB_DIR}")
        
        # Initialize database
        db_conn = connect(str(DB_PATH))
        init(db_conn)
        logger.info("Database initialized")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    initialize()

@app.on_event("shutdown")
async def shutdown_event():
    if db_conn:
        db_conn.close()
        logger.info("Database connection closed")

@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Detect faces
        faces = detector.detect(image)
        recognized_names = []
        confidences = []
        
        # Recognize faces
        for (x1, y1, x2, y2, conf) in faces:
            face = _prep_face(image, (x1, y1, x2, y2))
            if face is None:
                recognized_names.append(None)
                confidences.append(0.0)
                continue
            emb = embedder.embed_one(face)
            name, sim = match(emb, names, bank, sim_thresh=SIM_THRESH)
            recognized_names.append(name)
            confidences.append(sim)
            
            # Mark attendance if recognized
            if name:
                now = time.time()
                if name not in last_mark or (now - last_mark[name]) > MARK_COOLDOWN_SEC:
                    row = find_student_by_name(db_conn, name)
                    if row:
                        sid, name, group_id, section = row
                        mark_attendance(db_conn, sid, name, group_id, section, status="Present")
                        logger.info(f"Marked attendance for {name}")
                        last_mark[name] = now
        
        return DetectionResponse(
            faces=[(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf in faces],
            names=recognized_names,
            confidences=confidences
        )
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance/{date}")
async def get_attendance(date: str):
    try:
        cur = db_conn.cursor()
        cur.execute("SELECT id, name, group_id, section, ts, status FROM attendance WHERE date=? ORDER BY name", (date,))
        rows = cur.fetchall()
        return {
            "date": date,
            "records": [
                {"id": r[0], "name": r[1], "group_id": r[2], "section": r[3], "timestamp": r[4], "status": r[5]}
                for r in rows
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch attendance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_embeddings(dataset_dir: str):
    try:
        count = build_embeddings(dataset_dir, str(EMB_DIR), detector, conf_thresh=DET_CONF)
        global names, bank
        names, bank = load_bank(str(EMB_DIR))
        return {"message": f"Registered {count} identities"}
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```
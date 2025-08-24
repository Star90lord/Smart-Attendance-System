#
smart_attendance/
│
├── app.py                        # Main entry point (Flask or script runner)
│
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
│
├── cnn_model/                    # Face detection + recognition
│   ├── __init__.py
│   └── cnn_face_detector.py       # CNN-based face detector
│
├── database/                     # Database handling
│   ├── __init__.py
│   └── database_handler.py        # SQLite or MySQL handler
│
├── storage/                      # Storage for attendance data
│   ├── __init__.py
│   └── csv_handler.py             # CSV-based attendance logger
│               
├── static/                       # For frontend (if Flask UI is used)
│   ├── css/
│   ├── js/
│   └── images/
│
└── templates/                    # HTML templates (if Flask UI is used)
    └── index.htmls
#

# Smart Attendance System
A **Face Recognition-based Attendance System** built with **Python, OpenCV, and SQLite**.  
This project automatically detects and recognizes students' faces, marks their attendance in a database, and prevents duplicate entries for the same day.

--------------------------------------------------------------------------------------------------------------------------

## Features
- **Face Detection & Recognition** using OpenCV, CNN,DNN , MTCNN.
- **SQLite Database Integration** for storing:
  - Student details
  - Attendance records
- **Duplicate Prevention** – Attendance is marked only once per student per day.
- **Real-time Video Processing** from a webcam or camera feed.
- **Modular Code Structure** for easy maintenance and upgrades.
- **Export Attendance** to CSV for reports.

------------------------------------------------------------------------------------------------------------------------

## 🗄 Database Schema

### Table: `students`
| Column           | Type      | Description |
|------------------|-----------|-------------|
| id               | INTEGER PRIMARY KEY AUTOINCREMENT | Unique student ID |
| name             | TEXT      | Student name |
| roll_number      | TEXT UNIQUE | Roll number / ID |
| email            | TEXT      | Optional email |
| registered_face  | BLOB      | Optional face encoding |

### Table: `attendance`
| Column       | Type      | Description |
|--------------|-----------|-------------|
| id           | INTEGER PRIMARY KEY AUTOINCREMENT | Record ID |
| student_id   | INTEGER   | Linked to `students.id` |
| date         | TEXT      | Date (YYYY-MM-DD) |
| time         | TEXT      | Time (HH:MM:SS) |
| status       | TEXT      | Present / Absent |
| FOREIGN KEY  | (student_id) REFERENCES students(id) |

----------------------------------------------------------------------------------------------------------------------


### Important notes
- **YOLOv8 face weights**: use a face-specific `.pt` file (e.g., `yolov8n-face.pt`). General `yolov8n.pt` (COCO) won’t detect faces as a class. If you don’t have a face `.pt`, use the **SSD fallback** by supplying `--proto` and `--caffe_model`.
- This setup **runs on CPU**. For better FPS, reduce camera resolution or add frame skipping.
- Embeddings are stored as **one .npy per student** under `embeddings/` using the **student’s name**.
- During live run, only the **name** is drawn over the box, as you wanted.
- The database keeps one record per (student, date). Re-recognition within the cooldown window won’t spam.

You can now drop this folder into your project and run each part independently. If you want, I can also add **face alignment using landmarks** or a **Flask dashboard** next.
##
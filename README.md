# 🎯 Smart Attendance System

A **Face Recognition-based Attendance System** built with **Python, OpenCV, and SQLite**.  
This project automatically detects and recognizes students' faces, marks their attendance in a database, and prevents duplicate entries for the same day.

---

## 📌 Features
- **Face Detection & Recognition** using OpenCV.
- **SQLite Database Integration** for storing:
  - Student details
  - Attendance records
- **Duplicate Prevention** – Attendance is marked only once per student per day.
- **Real-time Video Processing** from a webcam or camera feed.
- **Modular Code Structure** for easy maintenance and upgrades.
- **Export Attendance** to CSV for reports.

---

## 📂 Project Structure

smart_attendance/
│
├── app.py # Main application entry point
├── database.py # SQLite database handling
├── face_recognition.py # Face detection & recognition logic
├── utils.py # Helper functions
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

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

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system

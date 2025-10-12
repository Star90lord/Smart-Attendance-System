from flask import Flask, render_template_string, request, jsonify, redirect
import cv2
import face_recognition
import base64
import pickle
import os
import sqlite3
import numpy as np
from datetime import datetime
from csv import DictReader

app = Flask(__name__)

DB_FILE = 'attendance.db'
PHOTOS_DIR = 'data'

def find_photo(name, photos_dir=PHOTOS_DIR):
    if not os.path.exists(photos_dir):
        return None
    first_name = name.split()[0]
    candidates = [
        f"{name}.jpg",
        f"{name}.jpeg",
        f"{name}.JPG",
        f"{name}.JPEG",
        f"{name.replace(' ', '')}.jpg",
        f"{name.replace(' ', '')}.jpeg",
        f"{name.replace(' ', '')}.JPG",
        f"{name.replace(' ', '')}.JPEG",
        f"{name.replace(' ', '_')}.jpg",
        f"{name.replace(' ', '_')}.jpeg",
        f"{name.replace(' ', '_')}.JPG",
        f"{name.replace(' ', '_')}.JPEG",
        f"{first_name}.jpg",
        f"{first_name}.jpeg",
        f"{first_name}.JPG",
        f"{first_name}.JPEG",
    ]
    for cand in candidates:
        path = os.path.join(photos_dir, cand)
        if os.path.exists(path):
            return path
    return None

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  roll_number TEXT UNIQUE,
                  email TEXT,
                  registered_face BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER,
                  date TEXT,
                  time TEXT,
                  status TEXT,
                  FOREIGN KEY(student_id) REFERENCES students (id))''')
    conn.commit()
    conn.close()

def init_from_csv():
    csv_path = 'details.csv'
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r') as f:
        reader = DictReader(f)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        inserted = 0
        for row in reader:
            name = row['NAMES'].strip()
            roll = row["SYSTEM ID'S"].strip()
            # Check if already exists
            c.execute("SELECT id FROM students WHERE roll_number = ?", (roll,))
            if c.fetchone():
                continue
            image_path = find_photo(name)
            if image_path:
                img = cv2.imread(image_path)
                if img is not None:
                    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encoding = face_recognition.face_encodings(rgb_image)
                    if encoding:
                        face_blob = pickle.dumps(encoding[0])
                        c.execute("INSERT INTO students (name, roll_number, email, registered_face) VALUES (?, ?, ?, ?)",
                                  (name, roll, '', face_blob))
                        inserted += 1
        conn.commit()
        conn.close()
    return inserted

init_db()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Smart Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-family: Arial, sans-serif;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card {
            border-radius: 20px;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
            background-color: #1e1e2f;
            padding: 30px;
            width: 600px;
            text-align: center;
        }
        .btn-custom {
            background-color: #667eea;
            border: none;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 12px;
            color: white;
            transition: 0.3s;
            margin: 5px;
        }
        .btn-custom:hover {
            background-color: #5765d3;
        }
        .result-box {
            background: #2d2d44;
            padding: 15px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .success { color: #4ade80; font-weight: bold; }
        .error { color: #f87171; font-weight: bold; }
        h2 { margin-bottom: 20px; color: white; }
        #video { width: 320px; height: 240px; border-radius: 10px; margin-top: 10px; }
        #canvas { display: none; }
        .hidden { display: none; }
        /* Spinner */
        .spinner {
            display: none;
            margin-top: 20px;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
    <script>
        let video = null;
        let stream = null;

        function initCamera() {
            video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(s => {
                    stream = s;
                    video.srcObject = stream;
                })
                .catch(err => console.error('Error accessing camera:', err));
        }

        function captureImage() {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 320, 240);
            const dataURL = canvas.toDataURL('image/jpeg');
            return dataURL.split(',')[1]; // base64 without prefix
        }

        function showSpinner(id) {
            document.getElementById(id).style.display = "block";
        }

        function hideSpinner(id) {
            document.getElementById(id).style.display = "none";
        }
    </script>
</head>
<body>
    <div class="card">
        <h2>📝 Smart Attendance System</h2>
        
        {% if count == 0 %}
            <div class="result-box mt-4">
                <p style="color: #facc15;">No students loaded. Initialize from CSV?</p>
                <a href="/init"><button type="button" class="btn-custom">Initialize Database</button></a>
            </div>
        {% endif %}
        
        <!-- Add Student Form -->
        <div id="addForm">
            <h5 style="color: white;">Add New Student</h5>
            <form action="/add" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" class="form-control mb-2" required>
                <input type="text" name="name" placeholder="Enter Name" class="form-control mb-2" required>
                <input type="text" name="roll_number" placeholder="Enter Roll Number" class="form-control mb-2" required>
                <input type="email" name="email" placeholder="Enter Email (optional)" class="form-control mb-2">
                <button type="submit" class="btn-custom">Add Student</button>
            </form>
        </div>

        <hr style="border-color: #667eea;">

        <!-- Mark Attendance -->
        <div id="attendDiv">
            <h5 style="color: white;">Mark Attendance</h5>
            <button type="button" class="btn-custom" onclick="initCamera(); document.getElementById('cameraDiv').classList.remove('hidden');">Start Camera</button>
            
            <div id="cameraDiv" class="hidden">
                <video id="video" autoplay></video>
                <br>
                <button type="button" class="btn-custom" onclick="markAttendance()">Capture & Mark</button>
                <button type="button" class="btn-custom" style="background-color: #f87171;" onclick="stopCamera()">Stop Camera</button>
            </div>
        </div>

        <!-- Spinner for Attendance -->
        <div id="attendSpinner" class="spinner">
            <div class="spinner-border text-warning" role="status">
                <span class="visually-hidden">Processing...</span>
            </div>
            <p style="margin-top:10px;">Processing face...</p>
        </div>

        {% if message %}
            <div class="result-box mt-4">
                <p class="{{ 'success' if success else 'error' }}">{{ message }}</p>
            </div>
        {% endif %}

        {% if recent_attendance %}
            <div class="result-box mt-4">
                <h5 style="color: white; margin-bottom: 10px;">Recent Attendance</h5>
                <ul style="text-align: left; color: #facc15;">
                    {% for entry in recent_attendance %}
                    <li>{{ entry }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% endif %}
    </div>

    <canvas id="canvas" width="320" height="240"></canvas>

    <script>
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            document.getElementById('cameraDiv').classList.add('hidden');
        }

        function markAttendance() {
            showSpinner('attendSpinner');
            const base64Image = captureImage();
            fetch('/recognize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: base64Image})
            })
            .then(response => response.json())
            .then(data => {
                hideSpinner('attendSpinner');
                alert(data.message);
                location.reload(); // Reload to show message
            })
            .catch(err => {
                hideSpinner('attendSpinner');
                alert('Error: ' + err);
            });
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    count = c.fetchone()[0]
    c.execute("""SELECT s.name, a.date, a.time FROM attendance a 
                 JOIN students s ON a.student_id = s.id 
                 ORDER BY a.id DESC LIMIT 5""")
    recent = [f"{row[0]} - {row[1]} {row[2]}" for row in c.fetchall()]
    conn.close()
    message = request.args.get('message')
    success = request.args.get('success') == 'true'
    return render_template_string(HTML_PAGE, message=message, success=success, recent_attendance=recent, count=count)

@app.route("/init")
def init_db_csv():
    inserted = init_from_csv()
    msg = f"Initialized {inserted} students from CSV" if inserted > 0 else "No new students added (check photos dir and CSV)"
    return redirect(f"/?message={msg}&success={'true' if inserted > 0 else 'false'}")

@app.route("/add", methods=["POST"])
def add_student():
    if 'image' not in request.files:
        return redirect("/?message=No image uploaded&success=false")
    
    file = request.files['image']
    name = request.form['name']
    roll_number = request.form['roll_number']
    email = request.form.get('email', '')
    
    if file.filename == '':
        return redirect("/?message=No image selected&success=false")
    
    # Read image
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(rgb_image)
    
    if not encoding:
        return redirect("/?message=No face detected&success=false")
    
    face_blob = pickle.dumps(encoding[0])
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (name, roll_number, email, registered_face) VALUES (?, ?, ?, ?)",
                  (name, roll_number, email, face_blob))
        conn.commit()
        return redirect("/?message=Student added successfully&success=true")
    except sqlite3.IntegrityError:
        return redirect("/?message=Roll number already exists&success=false")
    finally:
        conn.close()

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    base64_image = data['image']
    image_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_img)
    
    if not face_encodings:
        return jsonify({'message': 'No face detected'})
    
    test_encoding = face_encodings[0]
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, registered_face, name FROM students")
    rows = c.fetchall()
    
    matches = []
    for row in rows:
        student_id, face_blob, name = row
        known_encoding = pickle.loads(face_blob)
        match = face_recognition.compare_faces([known_encoding], test_encoding, tolerance=0.6)[0]
        if match:
            matches.append((student_id, name))
    
    if not matches:
        conn.close()
        return jsonify({'message': 'Unknown person'})
    
    # Take first match
    student_id, name = matches[0]
    
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?", (student_id, today))
    if c.fetchone()[0] > 0:
        conn.close()
        return jsonify({'message': f'{name} already marked today'})
    
    time_str = datetime.now().strftime("%H:%M:%S")
    c.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, 'Present')",
              (student_id, today, time_str))
    conn.commit()
    conn.close()
    return jsonify({'message': f'Attendance marked for {name}'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# MAIN.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
DATA_DIR = "data"
DETAILS_FILE = os.path.join(DATA_DIR, "details.vsv")  # simple text file (pipe-separated)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def safe_folder_name(name):
    # create a filesystem-safe folder name from the student's name
    # keep only alphanumerics, dash and underscore (replace others with underscore)
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return safe or "student"

def append_details_line(details_path, row_values):
    # row_values should be an iterable of strings (will be joined by '|')
    line = "|".join(row_values) + "\n"
    # append to the details.vsv file (utf-8)
    with open(details_path, "a", encoding="utf-8") as f:
        f.write(line)

@app.route("/api/enroll_student", methods=["POST"])
def enroll_student():
    """
    Expects multipart/form-data with:
      - student_photo (file)
      - student_id (string)
      - student_name (string)
      - course_id (string)
      - group_section (string)
    Saves:
      - data/details.vsv  (appends a pipe-separated line)
      - data/<student_folder>/<saved_image_file>
    Returns JSON:
      { ok: True/False, message: "...", ... }
    """
    # Required form fields
    student_id = request.form.get("student_id") or request.form.get("studentID") or ""
    student_name = request.form.get("student_name") or request.form.get("studentName") or ""
    course_id = request.form.get("course_id") or request.form.get("courseID") or ""
    group_section = request.form.get("group_section") or request.form.get("groupSection") or ""

    # Validate presence
    if not student_id or not student_name:
        return jsonify({"ok": False, "message": "Missing required fields: student_id and student_name are required."}), 400

    # File handling
    file = request.files.get("student_photo") or request.files.get("student_photo_file") or None
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "No photo file uploaded (field name: student_photo)."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "message": "File type not allowed. Allowed: png, jpg, jpeg, gif, webp"}), 400

    # Prepare student folder
    folder_name = safe_folder_name(student_name)
    student_folder = os.path.join(DATA_DIR, folder_name)
    os.makedirs(student_folder, exist_ok=True)

    # Construct a safe filename: roll_studentname_timestamp.ext
    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "jpg"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    saved_filename = f"{student_id}_{folder_name}_{timestamp}.{ext}"
    saved_path = os.path.join(student_folder, saved_filename)

    try:
        # Save image to disk
        file.save(saved_path)
    except Exception as e:
        return jsonify({"ok": False, "message": "Failed to save image", "error": str(e)}), 500

    # Prepare a simple record line and append to details.vsv
    # Format: timestamp | student_id | student_name | course_id | group_section | image_relative_path
    rel_image_path = os.path.join(folder_name, saved_filename).replace("\\", "/")
    record = [
        datetime.now().isoformat(),  # timestamp
        student_id,
        student_name,
        course_id,
        group_section,
        rel_image_path
    ]
    try:
        append_details_line(DETAILS_FILE, record)
    except Exception as e:
        # If writing details fails, try to remove saved image to avoid orphan files
        try:
            os.remove(saved_path)
        except Exception:
            pass
        return jsonify({"ok": False, "message": "Failed to write details file", "error": str(e)}), 500

    # Return JSON for frontend
    return jsonify({
        "ok": True,
        "message": "Enrollment saved",
        "student_id": student_id,
        "student_name": student_name,
        "course_id": course_id,
        "group_section": group_section,
        "image_path": rel_image_path
    }), 200

@app.route("/", methods=["GET"])
def index():
    return "Enrollment API running. POST to /api/enroll_student", 200

if __name__ == "__main__":
    # Run in production you should use a WSGI server. Debug on local only.
    app.run(host="0.0.0.0", port=5000, debug=True)

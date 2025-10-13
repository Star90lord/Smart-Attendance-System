# MAIN.py
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Configuration
DATA_DIR = "data"
DETAILS_CSV = os.path.join(DATA_DIR, "details.csv")  # use CSV as requested
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_folder_name(name):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (name or "").strip())
    return safe or "student"

def csv_has_headers(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(1024)
            f.seek(0)
            return sniffer.has_header(sample)
    except Exception:
        return False

def ensure_csv_exists(path):
    headers = ["timestamp", "student_id", "student_name", "course_id", "group_section", "image_path"]
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return
    # if file exists but has no header, add header by rewriting (safe guard)
    if not csv_has_headers(path):
        # read existing rows, rewrite with header
        try:
            with open(path, "r", newline="", encoding="utf-8") as fr:
                existing = list(csv.reader(fr))
        except Exception:
            existing = []
        with open(path, "w", newline="", encoding="utf-8") as fw:
            writer = csv.writer(fw)
            writer.writerow(headers)
            for row in existing:
                if row:
                    writer.writerow(row)

def read_existing_records(path):
    """
    Returns list of dicts representing existing rows.
    """
    records = []
    if not os.path.exists(path):
        return records
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize whitespace
            records.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return records

def is_duplicate(new_record, existing_records):
    """
    Determine duplicates:
     - If student_id present -> duplicate if any row has same student_id (case-insensitive).
     - Else -> duplicate if any row has same student_name + course_id + group_section.
    """
    sid = (new_record.get("student_id") or "").strip()
    name = (new_record.get("student_name") or "").strip().lower()
    course = (new_record.get("course_id") or "").strip().lower()
    group = (new_record.get("group_section") or "").strip().lower()

    if sid:
        for r in existing_records:
            if (r.get("student_id") or "").strip().lower() == sid.lower():
                return True
    else:
        for r in existing_records:
            if (r.get("student_id") or "").strip() == "":
                if ((r.get("student_name") or "").strip().lower() == name and
                    (r.get("course_id") or "").strip().lower() == course and
                    (r.get("group_section") or "").strip().lower() == group):
                    return True
    return False

@app.route("/api/enroll_student", methods=["POST"])
def enroll_student():
    # form fields (supporting multiple common names)
    student_id = (request.form.get("student_id") or request.form.get("studentID") or "").strip()
    student_name = (request.form.get("student_name") or request.form.get("studentName") or "").strip()
    course_id = (request.form.get("course_id") or request.form.get("courseID") or "").strip()
    group_section = (request.form.get("group_section") or request.form.get("groupSection") or "").strip()

    # Basic validation: require at least name or id
    if not student_id and not student_name:
        return jsonify({"ok": False, "message": "Provide at least student_id or student_name."}), 400

    # File handling
    file = request.files.get("student_photo") or request.files.get("student_photo_file") or None
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "No photo uploaded (field: student_photo)."}), 400
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "message": "File type not allowed."}), 400

    # Ensure CSV exists with header
    ensure_csv_exists(DETAILS_CSV)

    # Read existing records
    existing = read_existing_records(DETAILS_CSV)

    new_record = {
        "student_id": student_id,
        "student_name": student_name,
        "course_id": course_id,
        "group_section": group_section
    }

    # Duplicate check
    if is_duplicate(new_record, existing):
        return jsonify({"ok": False, "message": "Duplicate detected: student already exists in details.csv. No changes made."}), 200

    # Prepare student folder and filename
    folder_name = safe_folder_name(student_name or student_id)
    student_folder = os.path.join(DATA_DIR, folder_name)
    os.makedirs(student_folder, exist_ok=True)

    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit(".", 1)[1].lower()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    saved_filename = f"{student_id or 'noid'}_{folder_name}_{timestamp}.{ext}"
    saved_path = os.path.join(student_folder, saved_filename)

    try:
        file.save(saved_path)
    except Exception as e:
        return jsonify({"ok": False, "message": "Failed to save image", "error": str(e)}), 500

    # Append row to CSV
    image_rel_path = os.path.join(folder_name, saved_filename).replace("\\", "/")
    row = {
        "timestamp": datetime.now().isoformat(),
        "student_id": student_id,
        "student_name": student_name,
        "course_id": course_id,
        "group_section": group_section,
        "image_path": image_rel_path
    }
    try:
        with open(DETAILS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "student_id", "student_name", "course_id", "group_section", "image_path"])
            writer.writerow(row)
    except Exception as e:
        # cleanup image on failure
        try:
            os.remove(saved_path)
        except Exception:
            pass
        return jsonify({"ok": False, "message": "Failed to append to details.csv", "error": str(e)}), 500

    return jsonify({"ok": True, "message": "Student saved", **row}), 200

@app.route("/api/list_students", methods=["GET"])
def list_students():
    # returns all rows as JSON array of dicts
    ensure_csv_exists(DETAILS_CSV)
    records = read_existing_records(DETAILS_CSV)
    return jsonify(records), 200

@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    # serve images saved under data/<filename>
    return send_from_directory(DATA_DIR, filename, as_attachment=False)

@app.route("/", methods=["GET"])
def index():
    return "Enrollment API running. POST to /api/enroll_student", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

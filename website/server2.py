from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import mimetypes
import logging

app = Flask(__name__)

# Enable CORS for all routes (allows frontend to communicate from different origins)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DETAILS_FILE = os.path.join(DATA_DIR, "details.csv")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize CSV header if file doesn't exist
if not os.path.exists(DETAILS_FILE):
    with open(DETAILS_FILE, "w", encoding="utf-8") as f:
        f.write("timestamp|student_id|student_name|email|course_id|image_path\n")

def allowed_file(filename):
    """Check if file extension is allowed."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def safe_folder_name(name):
    """Create a filesystem-safe folder name from student name."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return safe or "student"

def append_details_line(details_path, row_values):
    """Append a pipe-separated line to details CSV file."""
    line = "|".join(str(v) for v in row_values) + "\n"
    try:
        with open(details_path, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except Exception as e:
        logger.error(f"Error writing to details file: {e}")
        return False

def validate_student_data(student_id, student_name, course_id=None):
    """Validate student data fields."""
    errors = []
    
    if not student_id or len(student_id.strip()) == 0:
        errors.append("student_id is required")
    
    if not student_name or len(student_name.strip()) == 0:
        errors.append("student_name is required")
    
    if student_id and len(student_id.strip()) > 20:
        errors.append("student_id must be less than 20 characters")
    
    if student_name and len(student_name.strip()) > 100:
        errors.append("student_name must be less than 100 characters")
    
    return errors

@app.route("/api/add_student", methods=["POST"])
def add_student():
    """
    Add a new student (from ew.html attendance system).
    Expects multipart/form-data with:
      - student_photo (file)
      - student_id (string)
      - student_name (string)
      - email (string, optional)
      - course_id (string)
    """
    try:
        # Extract form data
        student_id = request.form.get("student_id", "").strip()
        student_name = request.form.get("student_name", "").strip()
        email = request.form.get("email", "").strip() or "N/A"
        course_id = request.form.get("course_id", "").strip()
        student_photo = request.files.get("student_photo")

        # Validate data
        validation_errors = validate_student_data(student_id, student_name, course_id)
        if validation_errors:
            return jsonify({
                "success": False,
                "message": "Validation error",
                "errors": validation_errors
            }), 400

        # Validate file
        if not student_photo or student_photo.filename == "":
            return jsonify({
                "success": False,
                "message": "No photo file uploaded. Please select a student photo."
            }), 400

        if not allowed_file(student_photo.filename):
            return jsonify({
                "success": False,
                "message": f"File type not allowed. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Check file size
        student_photo.seek(0, os.SEEK_END)
        file_size = student_photo.tell()
        student_photo.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "message": f"File size exceeds limit of {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            }), 400

        # Create safe filename
        original_filename = secure_filename(student_photo.filename)
        ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{student_id}_{timestamp}.{ext}"
        saved_path = os.path.join(UPLOADS_DIR, saved_filename)

        # Save file
        try:
            student_photo.save(saved_path)
            logger.info(f"File saved: {saved_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to save image file",
                "error": str(e)
            }), 500

        # Append to CSV
        record = [
            datetime.now().isoformat(),
            student_id,
            student_name,
            email,
            course_id,
            saved_filename
        ]

        if not append_details_line(DETAILS_FILE, record):
            try:
                os.remove(saved_path)
            except Exception:
                pass
            return jsonify({
                "success": False,
                "message": "Failed to save student record to database"
            }), 500

        logger.info(f"Student added successfully: {student_id} - {student_name}")

        return jsonify({
            "success": True,
            "message": "Student added successfully",
            "student_id": student_id,
            "student_name": student_name,
            "email": email,
            "course": course_id,
            "image_path": saved_filename
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in add_student: {e}")
        return jsonify({
            "success": False,
            "message": "An unexpected error occurred",
            "error": str(e)
        }), 500


@app.route("/api/enroll_student", methods=["POST"])
def enroll_student():
    """
    Enroll a new student (from web.html enrollment system).
    Expects multipart/form-data with:
      - student_photo (file)
      - student_id (string)
      - student_name (string)
      - course_id (string)
      - group_section (string)
    """
    try:
        # Extract form data
        student_id = request.form.get("student_id", "").strip()
        student_name = request.form.get("student_name", "").strip()
        course_id = request.form.get("course_id", "").strip()
        group_section = request.form.get("group_section", "").strip()
        student_photo = request.files.get("student_photo")

        # Validate data
        validation_errors = validate_student_data(student_id, student_name, course_id)
        if validation_errors:
            return jsonify({
                "ok": False,
                "message": "Validation error",
                "error_details": ", ".join(validation_errors)
            }), 400

        if not group_section:
            return jsonify({
                "ok": False,
                "message": "group_section is required"
            }), 400

        # Validate file
        if not student_photo or student_photo.filename == "":
            return jsonify({
                "ok": False,
                "message": "No photo file uploaded"
            }), 400

        if not allowed_file(student_photo.filename):
            return jsonify({
                "ok": False,
                "message": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Check file size
        student_photo.seek(0, os.SEEK_END)
        file_size = student_photo.tell()
        student_photo.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "ok": False,
                "message": f"File size exceeds limit of {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            }), 400

        # Create safe filename
        original_filename = secure_filename(student_photo.filename)
        ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{student_id}_{timestamp}.{ext}"
        saved_path = os.path.join(UPLOADS_DIR, saved_filename)

        # Save file
        try:
            student_photo.save(saved_path)
            logger.info(f"Enrollment file saved: {saved_path}")
        except Exception as e:
            logger.error(f"Error saving enrollment file: {e}")
            return jsonify({
                "ok": False,
                "message": "Failed to save image file",
                "error": str(e)
            }), 500

        # Append to CSV
        record = [
            datetime.now().isoformat(),
            student_id,
            student_name,
            "N/A",  # email
            course_id,
            saved_filename
        ]

        if not append_details_line(DETAILS_FILE, record):
            try:
                os.remove(saved_path)
            except Exception:
                pass
            return jsonify({
                "ok": False,
                "message": "Failed to save enrollment record"
            }), 500

        logger.info(f"Student enrolled successfully: {student_id} - {student_name}")

        return jsonify({
            "ok": True,
            "message": "Enrollment saved successfully",
            "student_id": student_id,
            "student_name": student_name,
            "course_name": course_id,
            "group_section": group_section,
            "image_path": saved_filename
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in enroll_student: {e}")
        return jsonify({
            "ok": False,
            "message": "An unexpected error occurred",
            "error": str(e)
        }), 500


@app.route("/api/students", methods=["GET"])
def get_students():
    """Retrieve all enrolled students."""
    try:
        students = []
        if os.path.exists(DETAILS_FILE):
            with open(DETAILS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split("|")
                    if len(parts) >= 6:
                        students.append({
                            "timestamp": parts[0],
                            "student_id": parts[1],
                            "student_name": parts[2],
                            "email": parts[3],
                            "course_id": parts[4],
                            "image_path": parts[5]
                        })
        
        return jsonify({
            "success": True,
            "students": students,
            "total": len(students)
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving students: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to retrieve students",
            "error": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "server": "Attendance System API",
        "version": "1.0.0"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Welcome endpoint."""
    return jsonify({
        "message": "Attendance System API is running",
        "endpoints": {
            "POST /api/add_student": "Add student (from attendance system)",
            "POST /api/enroll_student": "Enroll student (from enrollment system)",
            "GET /api/students": "List all students",
            "GET /api/health": "Health check"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "message": "Endpoint not found",
        "error": str(error)
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False,
        "message": "Internal server error",
        "error": str(error)
    }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Attendance System API Server")
    print("=" * 60)
    print(f"Data Directory: {os.path.abspath(DATA_DIR)}")
    print(f"Uploads Directory: {os.path.abspath(UPLOADS_DIR)}")
    print(f"Details File: {os.path.abspath(DETAILS_FILE)}")
    print("=" * 60)
    print("Server running on:")
    print("  - http://127.0.0.1:5000")
    print("  - http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
## db/sql_store.py
import sqlite3
from datetime import datetime
import csv

SCHEMA_STUDENTS = (
    "CREATE TABLE IF NOT EXISTS students ("
    " id INTEGER PRIMARY KEY,"
    " name TEXT UNIQUE NOT NULL,"
    " group_id TEXT,"
    " section TEXT"
    ")"
)

SCHEMA_ATTENDANCE = (
    "CREATE TABLE IF NOT EXISTS attendance ("
    " id INTEGER,"
    " name TEXT,"
    " group_id TEXT,"
    " section TEXT,"
    " ts TEXT,"
    " date TEXT,"
    " status TEXT,"
    " PRIMARY KEY (id, date)"
    ")"
)


def connect(db_path: str):
    return sqlite3.connect(db_path)


def init(conn):
    cur = conn.cursor()
    cur.execute(SCHEMA_STUDENTS)
    cur.execute(SCHEMA_ATTENDANCE)
    conn.commit()


def import_students_csv(conn, csv_path: str):
    cur = conn.cursor()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['id'])
            name = row['name'].strip()
            group_id = row.get('group', row.get('group_id', '')).strip()
            section = row.get('section', '').strip()
            cur.execute(
                "INSERT OR IGNORE INTO students (id, name, group_id, section) VALUES (?, ?, ?, ?)",
                (sid, name, group_id, section)
            )
    conn.commit()


def mark_attendance(conn, sid: int, name: str, group: str, section: str, status: str = "Present"):
    cur = conn.cursor()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    cur.execute(
        "INSERT OR REPLACE INTO attendance (id, name, group_id, section, ts, date, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, name, group, section, ts, date_str, status)
    )
    conn.commit()


def find_student_by_name(conn, name: str):
    cur = conn.cursor()
    cur.execute("SELECT id, name, group_id, section FROM students WHERE name=?", (name,))
    return cur.fetchone()


def export_attendance(conn, for_date: str, out_csv: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id,name,group_id,section,date,status,ts FROM attendance WHERE date=? ORDER BY name", (for_date,))
    rows = cur.fetchall()
    import csv as _csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(["id", "name", "group", "section", "date", "status", "timestamp"])
        w.writerows(rows)
    return len(rows)
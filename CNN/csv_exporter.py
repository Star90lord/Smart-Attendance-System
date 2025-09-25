## reports/csv_exporter.py
from db.sql_store import export_attendance

def export_for_date(conn, date_str: str, out_csv: str):
    n = export_attendance(conn, date_str, out_csv)
    print(f"[✓] Exported {n} rows → {out_csv}")
import os
import csv
import pandas as pd
from datetime import datetime


ATTENDANCE_DIR  = "attendance"
ATTENDANCE_CSV  = os.path.join(ATTENDANCE_DIR, "attendance.csv")
ATTENDANCE_FILE = ATTENDANCE_CSV          # alias used by app.py
CSV_COLUMNS     = ["Name", "Date", "Time", "Status"]


def ensure_attendance_file():
    """Create attendance directory and CSV file if missing."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(os.path.join(ATTENDANCE_DIR, "reports"), exist_ok=True)

    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def load_attendance_df():
    """Load attendance CSV into a DataFrame."""
    ensure_attendance_file()
    try:
        df = pd.read_csv(ATTENDANCE_CSV)
        # Ensure columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def is_attendance_marked_today(name, date_str=None):
    """
    Check if attendance is already marked for a student today.
    Returns True if already marked.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    df = load_attendance_df()
    if df.empty:
        return False

    mask = (df["Name"] == name) & (df["Date"] == date_str)
    return mask.any()


def mark_attendance(name, status="Present"):
    """
    Mark attendance for a student.
    Returns:
        "marked"          – successfully marked
        "already_marked"  – duplicate for today
        "error"           – something went wrong
    """
    ensure_attendance_file()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        if is_attendance_marked_today(name, date_str):
            return "already_marked"

        with open(ATTENDANCE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str, status])

        return "marked"

    except Exception:
        return "error"


def get_attendance_for_date(date_str):
    """Return DataFrame filtered by date."""
    df = load_attendance_df()
    if df.empty:
        return df
    return df[df["Date"] == date_str].reset_index(drop=True)


def get_attendance_for_student(name):
    """Return DataFrame filtered by student name."""
    df = load_attendance_df()
    if df.empty:
        return df
    return df[df["Name"] == name].reset_index(drop=True)


def get_all_dates():
    """Return sorted list of unique dates in attendance CSV."""
    df = load_attendance_df()
    if df.empty or "Date" not in df.columns:
        return []
    return sorted(df["Date"].dropna().unique().tolist(), reverse=True)


def get_all_students():
    """Return sorted list of unique student names."""
    df = load_attendance_df()
    if df.empty or "Name" not in df.columns:
        return []
    return sorted(df["Name"].dropna().unique().tolist())


def get_attendance_summary():
    """Return summary statistics."""
    df = load_attendance_df()
    if df.empty:
        return {
            "total_records": 0,
            "unique_students": 0,
            "unique_dates": 0,
            "today_count": 0
        }

    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today] if "Date" in df.columns else pd.DataFrame()

    return {
        "total_records": len(df),
        "unique_students": df["Name"].nunique() if "Name" in df.columns else 0,
        "unique_dates": df["Date"].nunique() if "Date" in df.columns else 0,
        "today_count": len(today_df)
    }


def export_attendance_csv(date_str=None):
    """
    Return path to the attendance CSV.
    If date_str is given, exports filtered CSV to a temp file.
    """
    if date_str is None:
        return ATTENDANCE_CSV

    filtered = get_attendance_for_date(date_str)
    export_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    filtered.to_csv(export_path, index=False)
    return export_path

import sqlite3
import os

DB_PATH = "chronoguard.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    print(f"[DB] Initializing database at {DB_PATH}...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create cameras table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            face_blur_enabled INTEGER DEFAULT 0
        )
    """)
    
    # Create settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    # Seed default camera if table is empty
    cursor.execute("SELECT COUNT(*) FROM cameras")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO cameras (id, name, source, face_blur_enabled) VALUES (?, ?, ?, ?)",
            ("camera_1", "Default Webcam", "0", 0)
        )
        print("[DB] Seeded default camera_1 (webcam source '0')")
        
    # Seed default settings if empty
    cursor.execute("SELECT COUNT(*) FROM settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("retention_days", "30")
        )
        print("[DB] Seeded default retention_days (30 days)")
        
    conn.commit()
    conn.close()

def get_camera(camera_id: str) -> dict | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_cameras() -> list[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_camera_source(camera_id: str, source: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO cameras (id, name, source) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET source=?",
        (camera_id, camera_id, source, source)
    )
    conn.commit()
    conn.close()

def update_camera_blur(camera_id: str, enabled: bool):
    conn = get_db_connection()
    cursor = conn.cursor()
    val = 1 if enabled else 0
    cursor.execute(
        "UPDATE cameras SET face_blur_enabled = ? WHERE id = ?",
        (val, camera_id)
    )
    conn.commit()
    conn.close()

def get_setting(key: str, default: str = None) -> str | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row["value"]
    return default

def set_setting(key: str, value: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=?",
        (key, value, value)
    )
    conn.commit()
    conn.close()

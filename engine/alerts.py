import cv2
import time
import os
import json
from engine.db import insert_alert

class AlertSystem:
    def __init__(self, static_dir="static_snapshots"):
        self.static_dir = static_dir
        
        # Ensure directories exist
        os.makedirs(self.static_dir, exist_ok=True)

    def log_alert(self, alert_type, frame):
        """
        Saves a snapshot of the frame and logs the alert to the SQLite database.
        """
        timestamp = time.strftime("%H:%M:%S")
        timestamp_fs = time.strftime("%H_%M_%S") # File system safe
        
        # Save snapshot
        snapshot_filename = f"alert_{alert_type.replace(' ', '_')}_{timestamp_fs}.jpg"
        snapshot_path = os.path.join(self.static_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
        
        # Create alert record
        alert_id = int(time.time())
        snapshot_url = f"/static/{snapshot_filename}"
        alert_record = {
            "id": alert_id,
            "time": timestamp,
            "type": alert_type,
            "snapshot_url": snapshot_url
        }
        
        # Insert into SQLite Database
        try:
            insert_alert(alert_id, timestamp, alert_type, snapshot_url)
            print(f"[{timestamp}] ALERT SAVED TO DB: {alert_type}")
        except Exception as e:
            print(f"[{timestamp}] ERROR: Failed to save alert to DB: {e}")
            
        return alert_record

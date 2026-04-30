import cv2
import time
import os
import json

class AlertSystem:
    def __init__(self, static_dir="../dashboard/public/static", log_file="alerts.json"):
        self.static_dir = static_dir
        self.log_file = log_file
        
        # Ensure directories exist
        os.makedirs(self.static_dir, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log_alert(self, alert_type, frame):
        """
        Saves a snapshot of the frame and logs the alert to JSON.
        """
        timestamp = time.strftime("%H:%M:%S")
        timestamp_fs = time.strftime("%H_%M_%S") # File system safe
        
        # Save snapshot
        snapshot_filename = f"alert_{alert_type.replace(' ', '_')}_{timestamp_fs}.jpg"
        snapshot_path = os.path.join(self.static_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
        
        # Create alert record
        alert_record = {
            "id": int(time.time()),
            "time": timestamp,
            "type": alert_type,
            "snapshot_url": f"/static/{snapshot_filename}"
        }
        
        # Append to log
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
            
        logs.append(alert_record)
        
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=4)
            
        print(f"[{timestamp}] ALERT LOGGED: {alert_type}")
        return alert_record

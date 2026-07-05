import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import time
import json
import numpy as np
from fastapi import HTTPException

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.privacy_filter import PrivacyFilter
from engine.db import init_db, get_camera, update_camera_blur, get_setting, set_setting
from main import verify_staff_access, purge_old_alerts, STATIC_DIR

class TestPrivacySettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize test SQLite DB
        init_db()

    def test_verify_staff_access(self):
        # Valid credentials
        mock_cred = MagicMock()
        mock_cred.credentials = "ChronoGuardStaffToken2026"
        res = verify_staff_access(credentials=mock_cred)
        self.assertEqual(res, "staff")
        
        # Valid token via query param
        res2 = verify_staff_access(credentials=None, token="ChronoGuardStaffToken2026")
        self.assertEqual(res2, "staff")
        
        # Invalid credentials should raise HTTPException
        mock_bad_cred = MagicMock()
        mock_bad_cred.credentials = "bad_token"
        with self.assertRaises(HTTPException) as ctx:
            verify_staff_access(credentials=mock_bad_cred)
        self.assertEqual(ctx.exception.status_code, 401)
        
        # No credentials should raise HTTPException
        with self.assertRaises(HTTPException) as ctx:
            verify_staff_access(credentials=None, token=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_database_camera_toggle(self):
        # Default webcam should start as face_blur_enabled=0
        cam = get_camera("camera_1")
        self.assertIsNotNone(cam)
        self.assertEqual(cam["face_blur_enabled"], 0)
        
        # Toggle to True
        update_camera_blur("camera_1", True)
        cam = get_camera("camera_1")
        self.assertEqual(cam["face_blur_enabled"], 1)
        
        # Toggle back to False
        update_camera_blur("camera_1", False)
        cam = get_camera("camera_1")
        self.assertEqual(cam["face_blur_enabled"], 0)

    @patch('tracker.privacy_filter.cv2.FaceDetectorYN')
    def test_privacy_filter_redacts_faces(self, mock_yunet_class):
        # Mock FaceDetectorYN
        mock_detector = MagicMock()
        # Mock detection returning one face at coords [10, 20, 30, 40]
        # face output row shape: [x, y, w, h, ...]
        mock_detector.detect.return_value = (None, np.array([[10, 20, 30, 40, 0,0,0,0,0,0,0,0,0,0, 0.9]], dtype=np.float32))
        mock_yunet_class.create.return_value = mock_detector
        
        pf = PrivacyFilter()
        
        # Enable redaction
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 10
        # Mock GaussianBlur to return a distinct value so we can check if it ran
        with patch('tracker.privacy_filter.cv2.GaussianBlur') as mock_blur:
            mock_blur.return_value = np.ones((40, 30, 3), dtype=np.uint8) * 255
            
            blurred = pf.apply(frame, enabled=True)
            
            # Assert detect was called and GaussianBlur ran on target ROI
            mock_detector.detect.assert_called_once()
            mock_blur.assert_called_once()
            # Bounding box crop coordinates check
            self.assertEqual(mock_blur.call_args[0][0].shape, (40, 30, 3))

    def test_retention_purge_job(self):
        # Create a mock alert log
        log_file = "alerts.json"
        
        # Alert 0: 31 days old (expired if retention is 30 days)
        old_alert_id = int(time.time() - (31 * 24 * 3600))
        # Alert 1: 5 days old (not expired)
        new_alert_id = int(time.time() - (5 * 24 * 3600))
        
        # Create mock snapshot files
        old_snapshot = "alert_expired_test.jpg"
        new_snapshot = "alert_active_test.jpg"
        old_path = os.path.join(STATIC_DIR, old_snapshot)
        new_path = os.path.join(STATIC_DIR, new_snapshot)
        
        with open(old_path, "w") as f:
            f.write("fake_image_data")
        with open(new_path, "w") as f:
            f.write("fake_image_data")
            
        mock_logs = [
            {
                "id": old_alert_id,
                "time": "12:00:00",
                "type": "Expired Alert",
                "snapshot_url": f"/static/{old_snapshot}"
            },
            {
                "id": new_alert_id,
                "time": "12:05:00",
                "type": "Active Alert",
                "snapshot_url": f"/static/{new_snapshot}"
            }
        ]
        
        with open(log_file, "w") as f:
            json.dump(mock_logs, f, indent=4)
            
        # Set retention days to 30
        set_setting("retention_days", "30")
        
        # Run purge
        purge_old_alerts()
        
        # Verify old snapshot file is deleted, new remains
        self.assertFalse(os.path.exists(old_path))
        self.assertTrue(os.path.exists(new_path))
        
        # Verify log file has purged expired alert
        with open(log_file, "r") as f:
            active_logs = json.load(f)
            
        self.assertEqual(len(active_logs), 1)
        self.assertEqual(active_logs[0]["id"], new_alert_id)
        
        # Clean up
        if os.path.exists(new_path):
            os.remove(new_path)
        if os.path.exists(log_file):
            os.remove(log_file)

if __name__ == "__main__":
    unittest.main()

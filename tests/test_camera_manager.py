import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import time

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.camera_manager import CameraCaptureThread, CameraManager

class TestCameraManager(unittest.TestCase):
    @patch('tracker.camera_manager.cv2.VideoCapture')
    def test_camera_capture_thread_success(self, mock_vc_class):
        # Setup mock VideoCapture to succeed in opening and reading one frame
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.read.side_effect = [(True, "fake_frame"), (False, None)]
        mock_vc_class.return_value = mock_vc

        thread = CameraCaptureThread("cam_test", "0")
        
        # We patch time.sleep in the thread to run instantly
        with patch('tracker.camera_manager.time.sleep') as mock_sleep:
            # We run the loop once. To do this, we can let it run in a separate thread
            # or mock the self.running flag to terminate after one loop.
            thread.running = True
            
            # Manually run a single iteration of the outer loop to check behavior
            # We simulate the first iteration of the outer while self.running loop:
            # 1. cv2.VideoCapture()
            # 2. cap.isOpened() -> True
            # 3. cap.read() -> fake_frame
            # 4. cap.read() -> False (breaks inner loop)
            # 5. cap.release()
            # 6. We manually stop the thread inside the loop by setting running to False
            
            # Let's mock the sleep side effect to stop self.running
            def stop_running(*args, **kwargs):
                thread.running = False
            mock_sleep.side_effect = stop_running
            
            thread.run()
            
            self.assertEqual(thread.latest_frame, "fake_frame")
            self.assertTrue(mock_vc.release.called)

    @patch('tracker.camera_manager.cv2.VideoCapture')
    def test_reconnect_backoff(self, mock_vc_class):
        # Setup mock VideoCapture to fail opening
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = False
        mock_vc_class.return_value = mock_vc

        thread = CameraCaptureThread("cam_test", "rtsp://fake_url", max_backoff=8.0)
        
        # Test backoff exponential increments
        with patch('tracker.camera_manager.time.sleep') as mock_sleep:
            # We want to run the connection failure logic a few times.
            # We can capture the sleep durations passed to time.sleep.
            sleep_durations = []
            def record_sleep(secs):
                sleep_durations.append(secs)
                if len(sleep_durations) >= 4:
                    thread.running = False # Stop thread
            mock_sleep.side_effect = record_sleep
            
            thread.running = True
            thread.run()
            
            # Reconnect-with-backoff should double the sleep intervals: 1.0s -> 2.0s -> 4.0s -> 8.0s (max)
            self.assertEqual(sleep_durations, [1.0, 2.0, 4.0, 8.0])
            self.assertFalse(thread.is_connected)

    def test_camera_manager_lifecycle(self):
        # Mock CameraCaptureThread to avoid actually spawning threads
        with patch('tracker.camera_manager.CameraCaptureThread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.source = "0"
            mock_thread_class.return_value = mock_thread
            
            manager = CameraManager()
            
            # Register camera
            res = manager.register_camera("camera_1", "0")
            self.assertTrue(res)
            mock_thread.start.assert_called_once()
            
            # Register same camera again (should not re-initialize)
            res2 = manager.register_camera("camera_1", "0")
            self.assertTrue(res2)
            self.assertEqual(mock_thread.start.call_count, 1)
            
            # Register same camera with different source (should stop old and start new)
            mock_new_thread = MagicMock()
            mock_new_thread.source = "1"
            mock_thread_class.return_value = mock_new_thread
            
            res3 = manager.register_camera("camera_1", "1")
            self.assertTrue(res3)
            mock_thread.stop.assert_called_once()
            mock_new_thread.start.assert_called_once()

if __name__ == "__main__":
    unittest.main()

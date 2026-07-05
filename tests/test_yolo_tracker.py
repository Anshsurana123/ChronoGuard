import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.yolo_tracker import YoloTracker

class MockTensor:
    def __init__(self, val):
        self.val = np.array(val)
    def cpu(self):
        return self
    def numpy(self):
        return self.val

class MockBoxes:
    def __init__(self, xyxy, conf, ids, cls=None):
        self.xyxy = MockTensor(xyxy)
        self.conf = MockTensor(conf)
        self.id = MockTensor(ids) if ids is not None else None
        self.cls = MockTensor(cls if cls is not None else [0]*len(xyxy))

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes

class TestYoloTracker(unittest.TestCase):
    @patch('tracker.yolo_tracker.YOLOE')
    def setUp(self, mock_yolo_class):
        self.mock_model = MagicMock()
        mock_yolo_class.return_value = self.mock_model
        self.tracker = YoloTracker(confidence_threshold=0.3, grace_window=3)

    def test_init_tracking_containing_click(self):
        # Setup mock detections: two boxes
        # Box 0: [100, 100, 200, 200] (area 10000, ID 5)
        # Box 1: [50, 50, 250, 250] (area 40000, ID 8)
        # Click point: (150, 150) (inside both, should pick smallest)
        xyxy = [[100, 100, 200, 200], [50, 50, 250, 250]]
        conf = [0.8, 0.9]
        ids = [5, 8]
        
        mock_boxes = MockBoxes(xyxy, conf, ids)
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.tracker.init_tracking(frame, click_x=150, click_y=150)

        self.assertIsNotNone(res)
        self.assertEqual(self.tracker.tracked_id, 5) # Picked smallest area
        self.assertEqual(res["bbox"], [100.0, 100.0, 100.0, 100.0])
        self.assertEqual(res["centroid"], [150.0, 150.0])
        self.assertEqual(res["confidence"], 0.8)

    def test_init_tracking_fallback_closest_centroid(self):
        # Click point: (300, 300) (not inside any box)
        # Box 0: centroid (150, 150)
        # Box 1: centroid (280, 280) - closer to click point
        xyxy = [[100, 100, 200, 200], [260, 260, 300, 300]]
        conf = [0.8, 0.9]
        ids = [1, 2]
        
        mock_boxes = MockBoxes(xyxy, conf, ids)
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.tracker.init_tracking(frame, click_x=300, click_y=300)

        self.assertIsNotNone(res)
        self.assertEqual(self.tracker.tracked_id, 2) # Closer centroid

    def test_update_tracking_normal(self):
        # Initialize
        self.tracker.tracked_id = 5
        self.tracker.last_bbox = [100.0, 100.0, 100.0, 100.0]
        self.tracker.last_centroid = [150.0, 150.0]
        self.tracker.last_confidence = 0.8

        # Track update: Box 5 moves slightly
        xyxy = [[105, 105, 205, 205]]
        conf = [0.85]
        ids = [5]
        
        mock_boxes = MockBoxes(xyxy, conf, ids)
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.tracker.update_tracking(frame)

        self.assertIsNotNone(res)
        self.assertTrue(res["is_valid"])
        self.assertEqual(self.tracker.last_bbox, [105.0, 105.0, 100.0, 100.0])
        self.assertEqual(res["confidence"], 0.85)

    def test_runaway_bbox_guard_3x_area(self):
        # Initialize
        self.tracker.tracked_id = 5
        self.tracker.last_bbox = [100.0, 100.0, 100.0, 100.0] # area = 10000
        self.tracker.last_centroid = [150.0, 150.0]

        # Update: area jumps to 40000 (> 3x area jump)
        xyxy = [[50, 50, 250, 250]]
        conf = [0.8]
        ids = [5]
        
        mock_boxes = MockBoxes(xyxy, conf, ids)
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.tracker.update_tracking(frame)

        # Should be rejected but held at last-known position
        self.assertIsNotNone(res)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["bbox"], [100.0, 100.0, 100.0, 100.0])
        self.assertEqual(self.tracker.consecutive_rejected, 1)

        # If rejected repeatedly exceeding grace window (3), it should lose track
        for _ in range(3):
            res = self.tracker.update_tracking(frame)
        
        self.assertIsNone(res)
        self.assertIsNone(self.tracker.tracked_id)

    def test_runaway_bbox_guard_70percent(self):
        # Initialize
        self.tracker.tracked_id = 5
        self.tracker.last_bbox = [100.0, 100.0, 100.0, 100.0]
        self.tracker.last_centroid = [150.0, 150.0]

        # 640 * 480 = 307200 total area. 70% is 215040.
        # Box size: 600 * 400 = 240000 (exceeds 70%)
        xyxy = [[0, 0, 600, 400]]
        conf = [0.8]
        ids = [5]

        mock_boxes = MockBoxes(xyxy, conf, ids)
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.tracker.update_tracking(frame)

        self.assertIsNotNone(res)
        self.assertFalse(res["is_valid"])

    def test_lost_grace_window(self):
        # Initialize
        self.tracker.tracked_id = 5
        self.tracker.last_bbox = [100.0, 100.0, 100.0, 100.0]
        self.tracker.last_centroid = [150.0, 150.0]

        # Update: Target missing
        mock_boxes = MockBoxes([], [], [])
        self.mock_model.track.return_value = [MockResult(mock_boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Frame 1: Missing, should hold position but flag invalid
        res = self.tracker.update_tracking(frame)
        self.assertIsNotNone(res)
        self.assertFalse(res["is_valid"])
        self.assertEqual(self.tracker.consecutive_lost, 1)

        # Frame 2: Missing
        res = self.tracker.update_tracking(frame)
        self.assertIsNotNone(res)
        
        # Frame 3: Missing
        res = self.tracker.update_tracking(frame)
        self.assertIsNotNone(res)
        
        # Frame 4: Missing, exceeds grace_window (3), returns None and clears tracked_id
        res = self.tracker.update_tracking(frame)
        self.assertIsNone(res)
        self.assertIsNone(self.tracker.tracked_id)

if __name__ == "__main__":
    unittest.main()

import unittest
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.temporal_engine import TemporalEngine

class TestTemporalAnomaly(unittest.TestCase):
    def setUp(self):
        self.engine = TemporalEngine(time_hop_interval=5)
        # default confidence_threshold is 0.5

    def test_nominal_tracking(self):
        # When object is tracked and confidence is high, state is nominal
        state = self.engine.evaluate_tracking_state(is_tracked=True, confidence=0.8)
        self.assertEqual(state, "nominal")
        self.assertFalse(self.engine.is_backtracking)

    def test_anomaly_live_mode_low_confidence(self):
        # Live mode (forensic_mode = False)
        self.engine.forensic_mode = False
        
        # Test low confidence triggers live anomaly detection
        state = self.engine.evaluate_tracking_state(is_tracked=True, confidence=0.3)
        self.assertEqual(state, "anomaly_detected")
        
        # Test object lost triggers live anomaly detection
        state = self.engine.evaluate_tracking_state(is_tracked=False, confidence=0.8)
        self.assertEqual(state, "anomaly_detected")

    def test_anomaly_forensic_mode_backtracking(self):
        # Forensic mode (forensic_mode = True)
        self.engine.forensic_mode = True
        
        # Test low confidence triggers backtrack_triggered
        state = self.engine.evaluate_tracking_state(is_tracked=True, confidence=0.3)
        self.assertEqual(state, "backtrack_triggered")
        self.assertTrue(self.engine.is_backtracking)
        
        # Subsequent low confidence frames return "backtracking" state
        state = self.engine.evaluate_tracking_state(is_tracked=True, confidence=0.2)
        self.assertEqual(state, "backtracking")
        
        # Once re-acquired, goes to nominal/backtrack_complete
        state = self.engine.evaluate_tracking_state(is_tracked=True, confidence=0.8)
        self.assertEqual(state, "backtrack_complete")
        self.assertFalse(self.engine.is_backtracking)

if __name__ == "__main__":
    unittest.main()

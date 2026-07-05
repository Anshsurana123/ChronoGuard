class TemporalEngine:
    def __init__(self, time_hop_interval=15):
        """
        Initializes the Temporal Engine.
        :param time_hop_interval: Number of frames to skip in "Time-Hopping" Forensic Mode.
        """
        self.time_hop_interval = time_hop_interval
        self.frame_counter = 0
        self.forensic_mode = False
        
        # State tracking
        self.last_known_position = None
        self.confidence_threshold = 0.5
        self.is_backtracking = False

    def toggle_forensic_mode(self):
        self.forensic_mode = not self.forensic_mode
        return self.forensic_mode

    def should_process_frame(self):
        """
        Dynamic Frame Sampler logic. 
        Returns True if the frame should be sent to SAM 3 for inference.
        """
        self.frame_counter += 1
        
        # If we are backtracking due to an anomaly, we analyze every frame
        if self.is_backtracking:
            return True

        # In forensic mode, we skip frames (fast-forward) to speed up analysis
        if self.forensic_mode:
            return self.frame_counter % self.time_hop_interval == 0
            
        # In normal mode, process every frame (or could limit to e.g. 15 FPS)
        return True

    def evaluate_tracking_state(self, is_tracked, confidence):
        """
        Evaluates the tracking state.
        If object is lost (not tracked), confidence drops below threshold, or an anomaly is flagged,
        return an anomaly status.
        """
        is_anomaly = (not is_tracked) or (confidence < self.confidence_threshold)
        
        if is_anomaly:
            if self.forensic_mode:
                if not self.is_backtracking:
                    print("ANOMALY DETECTED: Object lost or confidence low. Triggering Backtracking...")
                    self.is_backtracking = True
                    return "backtrack_triggered"
                return "backtracking"
            else:
                print("ANOMALY DETECTED in live mode: tracking lost or low confidence.")
                return "anomaly_detected"
        else:
            # Successfully tracked, disable backtracking if it was on
            if self.is_backtracking:
                print("BACKTRACKING COMPLETE: Object re-acquired.")
                self.is_backtracking = False
                return "backtrack_complete"
                
        return "nominal"

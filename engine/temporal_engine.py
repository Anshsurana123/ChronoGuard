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
        Returns True if the frame should be sent to SAM 2 for inference.
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

    def evaluate_tracking_state(self, mask, confidence):
        """
        Evaluates the result from SAM 2. 
        If confidence drops unexpectedly during Time-Hopping, trigger backtracking.
        """
        if mask is None or confidence < self.confidence_threshold:
            if self.forensic_mode and not self.is_backtracking:
                print("ANOMALY DETECTED: Object lost or confidence low. Triggering Backtracking...")
                self.is_backtracking = True
                # In a real implementation, we would seek the video buffer backwards here.
                return "backtrack_triggered"
        else:
            # Successfully tracked, disable backtracking if it was on
            if self.is_backtracking:
                print("BACKTRACKING COMPLETE: Object re-acquired.")
                self.is_backtracking = False
                
        return "nominal"

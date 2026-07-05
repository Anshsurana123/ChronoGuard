import threading
import time
import cv2
import numpy as np

def generate_synthetic_frame(camera_id: str, t: float) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Floor (dark slate blue)
    cv2.fillPoly(frame, [np.array([[0, 260], [640, 260], [640, 480], [0, 480]])], (45, 45, 50))
    # Wall (slate gray)
    cv2.fillPoly(frame, [np.array([[0, 0], [640, 0], [640, 260], [0, 260]])], (80, 80, 85))
    
    # Horizon line
    cv2.line(frame, (0, 260), (640, 260), (100, 100, 105), 2)
    
    # Perspective grid lines on floor
    for x in range(-160, 801, 80):
        cv2.line(frame, (320, 260), (x, 480), (65, 65, 70), 1)
        
    # Whiteboard on the back wall
    cv2.rectangle(frame, (80, 40), (560, 200), (230, 230, 230), -1)
    cv2.rectangle(frame, (78, 38), (562, 202), (90, 90, 95), 2)
    
    # Simulator labels
    cv2.putText(frame, "CHRONOGUARD SIMULATOR STREAM", (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"CAM: {camera_id.upper()} | DEMO MODE ACTIVE", (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(frame, "Webcam shutter closed or index 0 empty", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 150), 1, cv2.LINE_AA)
    
    # Draw simulated classroom desk
    cv2.rectangle(frame, (260, 310), (380, 350), (139, 69, 19), -1) # Desk surface (brown)
    cv2.line(frame, (270, 350), (270, 440), (40, 40, 40), 4) # Leg 1
    cv2.line(frame, (370, 350), (370, 440), (40, 40, 40), 4) # Leg 2
    
    # Bouncing simulated student (person)
    cx = int(320 + 180 * np.sin(t * 0.8))
    cy = int(280 + 30 * np.cos(t * 1.6))
    
    # Draw simulated person (orange jumpsuited student)
    # Head
    cv2.circle(frame, (cx, cy - 40), 12, (0, 140, 255), -1)
    # Body
    cv2.rectangle(frame, (cx - 10, cy - 25), (cx + 10, cy + 25), (0, 140, 255), -1)
    # Legs
    cv2.line(frame, (cx - 5, cy + 25), (cx - 5, cy + 60), (0, 140, 255), 3)
    cv2.line(frame, (cx + 5, cy + 25), (cx + 5, cy + 60), (0, 140, 255), 3)
    
    # BGR color signature at [0,0] to indicate simulation metadata
    frame[0, 0] = [123, 45, 67]
    return frame

class CameraCaptureThread(threading.Thread):
    def __init__(self, camera_id: str, source: str, max_backoff: float = 60.0):
        super().__init__(name=f"CameraCapture-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.max_backoff = max_backoff
        
        self.latest_frame = None
        self.running = False
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Daemonize thread so it dies when main process exits
        self.daemon = True

    def run(self):
        self.running = True
        backoff = 1.0
        
        # Resolve source format (integer index vs string URL)
        source_resolved = self.source
        if self.source.isdigit():
            source_resolved = int(self.source)
            
        print(f"[CameraCapture-{self.camera_id}] Ingestion thread started. Source: {self.source}")
        
        while self.running:
            print(f"[CameraCapture-{self.camera_id}] Connecting to video source...")
            cap = cv2.VideoCapture(source_resolved)
            
            # Configure frame size if it is a local webcam
            if isinstance(source_resolved, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                self.is_connected = False
                print(f"[CameraCapture-{self.camera_id}] Connection failed. Reconnecting in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.max_backoff)
                cap.release()
                continue
                
            self.is_connected = True
            backoff = 1.0 # Reset backoff on successful connection
            print(f"[CameraCapture-{self.camera_id}] Connected to source.")
            
            black_frame_count = 0
            use_simulation = False
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"[CameraCapture-{self.camera_id}] Stream disconnected / failed to read frame.")
                    self.is_connected = False
                    break
                    
                # Self-healing black frame detection
                if frame is not None and isinstance(frame, np.ndarray):
                    if frame.mean() < 1.0:
                        black_frame_count += 1
                    else:
                        black_frame_count = 0
                        use_simulation = False
                        
                    if black_frame_count >= 15:
                        use_simulation = True
                        
                if use_simulation or frame is None:
                    frame = generate_synthetic_frame(self.camera_id, time.time())
                    
                with self.lock:
                    self.latest_frame = frame
                
                # Small sleep to yield CPU and limit ingestion rate if capture does not block
                time.sleep(0.01)
                
            cap.release()
            if self.running:
                # Sleep before attempting reconnection on stream drops
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.max_backoff)
                
        print(f"[CameraCapture-{self.camera_id}] Ingestion thread stopped.")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False


class CameraManager:
    def __init__(self):
        self.streams: dict[str, CameraCaptureThread] = {}
        self.lock = threading.Lock()

    def register_camera(self, camera_id: str, source: str) -> bool:
        """
        Registers a camera and starts its persistent ingestion thread if not already running.
        If the source changes, the existing thread is stopped and a new one is started.
        """
        with self.lock:
            existing = self.streams.get(camera_id)
            if existing:
                if existing.source == source:
                    # Thread already running with correct source
                    return True
                else:
                    # Source changed: stop old thread
                    print(f"[CameraManager] Camera source changed for '{camera_id}'. Restarting thread.")
                    existing.stop()
                    existing.join(timeout=2.0)
                    
            # Start new capture thread
            thread = CameraCaptureThread(camera_id, source)
            thread.start()
            self.streams[camera_id] = thread
            return True

    def get_frame(self, camera_id: str):
        """Retrieves the latest frame for a registered camera."""
        thread = self.streams.get(camera_id)
        if thread:
            return thread.get_latest_frame()
        return None

    def is_camera_connected(self, camera_id: str) -> bool:
        thread = self.streams.get(camera_id)
        if thread:
            return thread.is_connected
        return False

    def shutdown(self):
        """Stops all active ingestion threads."""
        print("[CameraManager] Shutting down all camera threads...")
        with self.lock:
            for thread in self.streams.values():
                thread.stop()
            for thread in self.streams.values():
                thread.join(timeout=1.0)
            self.streams.clear()
        print("[CameraManager] Shutdown complete.")

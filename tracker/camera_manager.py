import threading
import time
import cv2

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
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"[CameraCapture-{self.camera_id}] Stream disconnected / failed to read frame.")
                    self.is_connected = False
                    break
                    
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

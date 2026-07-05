import cv2
import os
import urllib.request
import numpy as np

class PrivacyFilter:
    def __init__(self, model_path: str = "face_detection_yunet_2023mar.onnx"):
        self.model_path = model_path
        self.enabled = True
        
        # Download YuNet ONNX model if not exists
        YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        if not os.path.exists(self.model_path):
            print(f"[PrivacyFilter] Downloading YuNet face detector model to {self.model_path}...")
            try:
                # Ensure parent directory exists (if any)
                os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
                urllib.request.urlretrieve(YUNET_URL, self.model_path)
                print("[PrivacyFilter] Download completed.")
            except Exception as e:
                print(f"[PrivacyFilter] Error downloading YuNet model: {e}")
                
        # Attempt to load YuNet
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(640, 480),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            print("[PrivacyFilter] YuNet face detector loaded successfully.")
        except Exception as e:
            print(f"[PrivacyFilter] ERROR: Failed to load YuNet face detector: {e}. Falling back to Haar Cascade.")
            self.detector = None
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def apply(self, frame, enabled: bool = True):
        """Detects faces and applies a heavy Gaussian blur to redact them."""
        if not enabled:
            return frame

        h, w = frame.shape[:2]
        
        if self.detector is not None:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        fx, fy, fw, fh = map(int, face[:4])
                        # Safeguard boundary constraints
                        fx = max(0, fx)
                        fy = max(0, fy)
                        fw = min(fw, w - fx)
                        fh = min(fh, h - fy)
                        if fw <= 0 or fh <= 0:
                            continue
                        
                        roi = frame[fy:fy+fh, fx:fx+fw]
                        
                        # Set blur kernel sizes relative to ROI size, limited to max 99
                        ksize = 99
                        kw = ksize if ksize < fw else (fw - 1 if fw % 2 == 0 else fw)
                        kh = ksize if ksize < fh else (fh - 1 if fh % 2 == 0 else fh)
                        
                        # Ensure kernel dimensions are odd and positive
                        if kw % 2 == 0: kw -= 1
                        if kh % 2 == 0: kh -= 1
                        kw = max(1, kw)
                        kh = max(1, kh)
                        
                        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 30)
                        frame[fy:fy+fh, fx:fx+fw] = blurred_roi
            except Exception as e:
                print(f"[PrivacyFilter] YuNet face detection failed: {e}. Falling back to Haar.")
                self._apply_haar(frame)
        else:
            self._apply_haar(frame)

        return frame

    def _apply_haar(self, frame):
        """Haar Cascade face blurring fallback."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            h, w = frame.shape[:2]
            for (fx, fy, fw, fh) in faces:
                fx = max(0, fx)
                fy = max(0, fy)
                fw = min(fw, w - fx)
                fh = min(fh, h - fy)
                if fw <= 0 or fh <= 0:
                    continue
                roi = frame[fy:fy+fh, fx:fx+fw]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                frame[fy:fy+fh, fx:fx+fw] = blurred_roi
        except Exception as e:
            print(f"[PrivacyFilter] Haar Cascade fallback failed: {e}")

    def toggle(self):
        """Toggles the global privacy filter on or off (legacy support)."""
        self.enabled = not self.enabled
        return self.enabled

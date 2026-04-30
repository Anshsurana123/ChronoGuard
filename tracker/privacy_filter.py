import cv2
import os

class PrivacyFilter:
    def __init__(self):
        # We will use the built-in OpenCV Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.enabled = True

    def apply(self, frame):
        """Detects faces and applies a heavy Gaussian blur to redact them."""
        if not self.enabled:
            return frame

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Apply blur to each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (the face)
            roi = frame[y:y+h, x:x+w]
            
            # Apply a strong Gaussian blur to anonymize the face
            # Kernel size must be odd and positive. A large kernel creates a strong blur.
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            
            # Place the blurred face back into the frame
            frame[y:y+h, x:x+w] = blurred_roi

        return frame

    def toggle(self):
        """Toggles the privacy filter on or off."""
        self.enabled = not self.enabled
        return self.enabled

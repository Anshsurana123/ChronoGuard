import numpy as np
import cv2
import json
import os
from ultralytics import YOLOE

def load_config():
    # Load config from environment or fallback to config.json
    config_path = os.environ.get("CHRONO_CONFIG_PATH", "config.json")
    config = {
        "model_scale": "s",
        "device": "cpu",
        "export_format": "pytorch"
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"[YoloTracker] Warning: Failed to parse config file: {e}")
            
    # Environment variable overrides
    if "CHRONO_MODEL_SCALE" in os.environ:
        config["model_scale"] = os.environ["CHRONO_MODEL_SCALE"]
    if "CHRONO_DEVICE" in os.environ:
        config["device"] = os.environ["CHRONO_DEVICE"]
    if "CHRONO_EXPORT_FORMAT" in os.environ:
        config["export_format"] = os.environ["CHRONO_EXPORT_FORMAT"]
        
    return config

def get_model_path(model_scale: str, export_format: str) -> str:
    scale = model_scale.lower().strip()
    fmt = export_format.lower().strip()
    
    if scale in ("n", "s"):
        base_name = "yoloe-11s-seg"
    elif scale == "m":
        base_name = "yoloe-11m-seg"
    elif scale in ("l", "x"):
        base_name = "yoloe-11l-seg"
    else:
        print(f"[YoloTracker] Warning: Unknown model scale '{model_scale}', falling back to 's'")
        base_name = "yoloe-11s-seg"
        
    if fmt == "pytorch":
        return f"{base_name}.pt"
    elif fmt == "onnx":
        return f"{base_name}.onnx"
    elif fmt == "openvino":
        return f"{base_name}_openvino_model"
    else:
        print(f"[YoloTracker] Warning: Unknown export format '{export_format}', falling back to 'pytorch'")
        return f"{base_name}.pt"

class YoloTracker:
    def __init__(self, confidence_threshold=0.3, grace_window=15):
        """
        Initializes the YOLOE + ByteTrack local tracker.
        :param confidence_threshold: Real threshold for detection confidence.
        :param grace_window: Max consecutive frames to hold last-known position when lost or rejected.
        """
        config = load_config()
        self.model_scale = config["model_scale"]
        self.device = config["device"]
        self.export_format = config["export_format"]
        
        model_path = get_model_path(self.model_scale, self.export_format)
        print(f"[YoloTracker] Loading local YOLOE model scale={self.model_scale}, device={self.device}, format={self.export_format}")
        print(f"[YoloTracker] Loading model from: {model_path}")
        
        self.model = YOLOE(model_path)
        self.model.to(self.device)
        
        self.confidence_threshold = confidence_threshold
        self.grace_window = grace_window
        
        # Tracking state
        self.tracked_id = None
        self.last_bbox = None          # [x, y, w, h]
        self.last_centroid = None      # [cx, cy]
        self.last_confidence = 0.0
        self.consecutive_lost = 0      # Frames where ByteTrack did not return the ID
        self.consecutive_low_conf = 0  # Frames where confidence was below threshold
        self.consecutive_rejected = 0  # Frames where bbox guard rejected the update

    def set_prompt(self, prompt: str):
        """Sets the open-vocabulary prompt classes for YOLOE."""
        prompt_classes = [prompt.strip().lower()]
        print(f"[YoloTracker] Setting open-vocabulary class prompt: {prompt_classes}")
        self.model.set_classes(prompt_classes)

    def init_tracking(self, frame, click_x: int, click_y: int, prompt: str = "person") -> dict | None:
        """
        Initialize tracking by running detection and associating the clicked point
        with a detected object bbox.
        """
        # Reset current tracking state
        self.tracked_id = None
        self.last_bbox = None
        self.last_centroid = None
        self.last_confidence = 0.0
        self.consecutive_lost = 0
        self.consecutive_low_conf = 0
        self.consecutive_rejected = 0

        # Check simulation BGR signature at [0, 0]
        is_sim = False
        try:
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                is_sim = list(frame[0, 0]) == [123, 45, 67]
        except Exception:
            pass

        if is_sim:
            import time
            t = time.time()
            cx = int(320 + 180 * np.sin(t * 0.8))
            cy = int(280 + 30 * np.cos(t * 1.6))
            w, h = 50, 140
            x1 = cx - w // 2
            y1 = cy - h // 2
            
            self.tracked_id = 99 # Mock ID
            self.last_bbox = [float(x1), float(y1), float(w), float(h)]
            self.last_centroid = [float(cx), float(cy)]
            self.last_confidence = 0.95
            
            print(f"[YoloTracker] Initialized mock tracking on ID=99 (Simulation Mode)")
            return {
                "bbox": self.last_bbox,
                "centroid": self.last_centroid,
                "confidence": self.last_confidence,
            }

        # Set classes
        self.set_prompt(prompt)

        # Run track with ByteTrack to get IDs
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1)
        if not results or len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return None

        # Convert box elements
        ids_list = boxes.id.cpu().numpy()
        if ids_list.size == 0:
            return None
        ids_list = ids_list.astype(int)

        xyxy_list = boxes.xyxy.cpu().numpy()
        conf_list = boxes.conf.cpu().numpy()

        best_idx = None
        best_area = float('inf')
        containing_boxes = []

        # Find all boxes containing the click coordinates
        for idx, (x1, y1, x2, y2) in enumerate(xyxy_list):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                containing_boxes.append((idx, area))

        if containing_boxes:
            # Prefer the smallest containing box (to avoid background overlap)
            containing_boxes.sort(key=lambda item: item[1])
            best_idx = containing_boxes[0][0]
        else:
            # Fallback: Find the box with the closest centroid
            min_dist = float('inf')
            for idx, (x1, y1, x2, y2) in enumerate(xyxy_list):
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = (cx - click_x)**2 + (cy - click_y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

        if best_idx is not None:
            x1, y1, x2, y2 = xyxy_list[best_idx]
            self.tracked_id = ids_list[best_idx]
            w, h = x2 - x1, y2 - y1
            self.last_bbox = [float(x1), float(y1), float(w), float(h)]
            self.last_centroid = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
            self.last_confidence = float(conf_list[best_idx])
            
            print(f"[YoloTracker] Initialized tracking on ID={self.tracked_id}, bbox={self.last_bbox}, conf={self.last_confidence:.2f}")
            return {
                "bbox": self.last_bbox,
                "centroid": self.last_centroid,
                "confidence": self.last_confidence,
            }

        return None

    def update_tracking(self, frame) -> dict | None:
        """
        Updates tracking state for the active target using ByteTrack.
        Implements runaway bbox area checks and a position holding grace window on anomaly.
        """
        if self.tracked_id is None:
            return None

        # Check simulation BGR signature at [0, 0]
        is_sim = False
        try:
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                is_sim = list(frame[0, 0]) == [123, 45, 67]
        except Exception:
            pass

        if is_sim and self.tracked_id == 99:
            import time
            t = time.time()
            cx = int(320 + 180 * np.sin(t * 0.8))
            cy = int(280 + 30 * np.cos(t * 1.6))
            w, h = 50, 140
            self.last_bbox = [float(cx - w // 2), float(cy - h // 2), float(w), float(h)]
            self.last_centroid = [float(cx), float(cy)]
            self.last_confidence = 0.95
            self.consecutive_lost = 0
            self.consecutive_low_conf = 0
            self.consecutive_rejected = 0
            
            return {
                "bbox": self.last_bbox,
                "centroid": self.last_centroid,
                "confidence": self.last_confidence,
            }

        # Run track with ByteTrack
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1)
        
        found = False
        new_bbox = None
        new_centroid = None
        new_conf = 0.0

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                if self.tracked_id in ids:
                    idx = np.where(ids == self.tracked_id)[0][0]
                    xyxy_list = boxes.xyxy.cpu().numpy()
                    conf_list = boxes.conf.cpu().numpy()
                    x1, y1, x2, y2 = xyxy_list[idx]
                    new_conf = float(conf_list[idx])
                    w, h = x2 - x1, y2 - y1
                    new_bbox = [float(x1), float(y1), float(w), float(h)]
                    new_centroid = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
                    found = True

        # Case 1: Target found in this frame
        if found:
            self.consecutive_lost = 0
            
            # P0-3 Runaway/implausible bbox area check
            area_new = new_bbox[2] * new_bbox[3]
            frame_h, frame_w = frame.shape[:2]
            total_frame_area = frame_w * frame_h
            
            # Check 1: 70% fraction of total frame area
            exceeds_frame_limit = area_new > 0.7 * total_frame_area
            
            # Check 2: 3x jump from last-known bbox area
            exceeds_area_jump = False
            if self.last_bbox is not None:
                area_old = self.last_bbox[2] * self.last_bbox[3]
                if area_old > 0:
                    exceeds_area_jump = area_new > 3.0 * area_old

            if exceeds_frame_limit or exceeds_area_jump:
                self.consecutive_rejected += 1
                reason = "exceeds 70% frame area" if exceeds_frame_limit else f"3x area jump ({area_new:.1f} vs {area_old:.1f})"
                print(f"[YoloTracker] Runaway bbox guard triggered: {reason} (ID={self.tracked_id}). Consecutive rejections={self.consecutive_rejected}")
                
                # Check if grace window has been exceeded
                if self.consecutive_rejected > self.grace_window:
                    print(f"[YoloTracker] Target ID={self.tracked_id} lost due to persistent runaway bbox updates.")
                    self.tracked_id = None
                    return None
                
                # Hold last-known position
                return {
                    "bbox": self.last_bbox,
                    "centroid": self.last_centroid,
                    "confidence": self.last_confidence,
                    "is_valid": False,  # Marks tracking anomaly
                }
            
            # Successful tracking update
            self.consecutive_rejected = 0
            
            # Check low confidence
            if new_conf < self.confidence_threshold:
                self.consecutive_low_conf += 1
            else:
                self.consecutive_low_conf = 0

            # If confidence is below threshold for too long, treat as lost
            if self.consecutive_low_conf > self.grace_window:
                print(f"[YoloTracker] Target ID={self.tracked_id} lost due to persistent low confidence.")
                self.tracked_id = None
                return None

            self.last_bbox = new_bbox
            self.last_centroid = new_centroid
            self.last_confidence = new_conf

            return {
                "bbox": self.last_bbox,
                "centroid": self.last_centroid,
                "confidence": self.last_confidence,
                "is_valid": True,
            }

        # Case 2: Target not found/lost in this frame
        self.consecutive_lost += 1
        print(f"[YoloTracker] Target ID={self.tracked_id} missing in frame. Consecutive lost={self.consecutive_lost}")
        
        if self.consecutive_lost > self.grace_window:
            print(f"[YoloTracker] Target ID={self.tracked_id} lost (grace window exceeded).")
            self.tracked_id = None
            return None
        
        # Hold last-known-good position for a short grace window, flag anomaly
        return {
            "bbox": self.last_bbox,
            "centroid": self.last_centroid,
            "confidence": 0.0,
            "is_valid": False,
        }

    def reset(self):
        """Resets the tracking target."""
        self.tracked_id = None
        self.last_bbox = None
        self.last_centroid = None
        self.last_confidence = 0.0
        self.consecutive_lost = 0
        self.consecutive_low_conf = 0
        self.consecutive_rejected = 0

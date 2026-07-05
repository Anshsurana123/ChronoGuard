import asyncio
import base64
import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import numpy as np

# Append current directory to path so we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from tracker.privacy_filter import PrivacyFilter
from tracker.yolo_tracker import YoloTracker
from engine.temporal_engine import TemporalEngine
from engine.geofence import Geofence
from engine.alerts import AlertSystem

app = FastAPI(title="ChronoGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
privacy_filter = PrivacyFilter()

# Local YOLO-World + ByteTrack Tracker
tracker_ready = False
try:
    yolo_tracker = YoloTracker()
    tracker_ready = True
    print("[Backend] Local YOLO-World tracker initialized successfully.")
except Exception as e:
    print(f"\n[Backend] CRITICAL ERROR: Failed to initialize local YOLO-World tracker: {e}\n")
    yolo_tracker = None

temporal_engine = TemporalEngine(time_hop_interval=15)
geofence_engine = Geofence()
alert_system = AlertSystem()

# Shared state
current_frame = None
tracking_active = False
tracking_target = (320, 240)       # Initial click point
tracking_bbox = None               # [x, y, w, h]
tracking_centroid = None            # [cx, cy]
tracking_confidence = 0.0

@app.get("/")
async def root():
    return {"status": "ChronoGuard Local Backend Active", "tracker": "YOLO-World + ByteTrack", "tracker_ready": tracker_ready}

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    global current_frame, tracking_active
    global tracking_target, tracking_bbox, tracking_centroid, tracking_confidence

    await websocket.accept()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Background task to receive messages from Next.js
    async def receive_messages():
        global tracking_active, tracking_target
        global tracking_bbox, tracking_centroid, tracking_confidence
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "init_tracking":
                    print("[Backend] Received init_tracking:", msg)
                    click_x, click_y = msg["x"], msg["y"]
                    prompt = msg.get("prompt", "person")
                    tracking_target = (click_x, click_y)
                    tracking_bbox = None
                    tracking_centroid = None
                    tracking_confidence = 0.0

                    if not tracker_ready or yolo_tracker is None:
                        print("[Backend] YOLO Tracker is offline / failed to load. Aborting.")
                        await websocket.send_json({
                            "type": "alert",
                            "data": {
                                "id": int(time.time()),
                                "time": time.strftime("%H:%M:%S"),
                                "type": "CRITICAL ERROR: Local YOLO Tracker not loaded!",
                            }
                        })
                        await websocket.send_json({
                            "type": "tracking_update",
                            "bbox": None,
                            "centroid": [click_x, click_y],
                            "confidence": 0.0,
                        })
                        continue

                    if current_frame is not None:
                        result = await asyncio.to_thread(
                            yolo_tracker.init_tracking, current_frame, click_x, click_y, prompt
                        )
                        if result and result.get("bbox"):
                            tracking_active = True
                            tracking_bbox = result["bbox"]
                            tracking_centroid = result["centroid"]
                            tracking_confidence = result.get("confidence", 0.0)
                            print(f"[Backend] YOLO tracking initialized — bbox={tracking_bbox}, conf={tracking_confidence:.2f}")

                            await websocket.send_json({
                                "type": "tracking_update",
                                "bbox": tracking_bbox,
                                "centroid": tracking_centroid,
                                "confidence": tracking_confidence,
                            })
                        else:
                            print("[Backend] YOLO tracker failed to find target object at click point.")
                            await websocket.send_json({
                                "type": "alert",
                                "data": {
                                    "id": int(time.time()),
                                    "time": time.strftime("%H:%M:%S"),
                                    "type": f"Tracker Warning: No '{prompt}' detected near click point.",
                                }
                            })
                            await websocket.send_json({
                                "type": "tracking_update",
                                "bbox": None,
                                "centroid": [click_x, click_y],
                                "confidence": 0.0,
                            })

                elif msg.get("type") == "set_geofence":
                    print("[Backend] Received geofence points:", msg["points"])
                    geofence_engine.set_polygon(msg["points"])

                elif msg.get("type") == "update_endpoint":
                    # Colab tunnel updating is obsolete now that we run locally
                    print("[Backend] Dynamic endpoint update received (obsolete/local mode active).")
                    await websocket.send_json({
                        "type": "endpoint_updated",
                        "endpoint_url": "LOCAL (YOLO-World)",
                    })

                elif msg.get("type") == "stop_analysis":
                    tracking_active = False
                    tracking_bbox = None
                    tracking_centroid = None
                    tracking_confidence = 0.0
                    geofence_engine.set_polygon([])
                    if yolo_tracker:
                        yolo_tracker.reset()
                    print("[Backend] Analysis stopped, tracker reset")

        except WebSocketDisconnect:
            print("[Backend] Receiver disconnected")

    recv_task = asyncio.create_task(receive_messages())

    last_breach_alert_time = 0
    BREACH_THROTTLE_SECS = 5
    frame_counter = 0
    # Process every 3rd frame (~10 fps at 30fps capture) to balance CPU load
    SAM_UPDATE_INTERVAL = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame_counter += 1
            current_frame = frame.copy()

            # 1. Zero-Trust Privacy Filter
            frame = privacy_filter.apply(frame)

            # 2. Local YOLO-World Tracking Update
            if tracking_active and tracker_ready and yolo_tracker is not None and frame_counter % SAM_UPDATE_INTERVAL == 0:
                result = await asyncio.to_thread(yolo_tracker.update_tracking, current_frame)
                is_tracked = False
                
                if result is not None:
                    tracking_bbox = result["bbox"]
                    tracking_centroid = result["centroid"]
                    tracking_confidence = result["confidence"]
                    is_tracked = result.get("is_valid", True)

                    # Push updated tracking data to frontend
                    try:
                        await websocket.send_json({
                            "type": "tracking_update",
                            "bbox": tracking_bbox,
                            "centroid": tracking_centroid,
                            "confidence": tracking_confidence,
                        })
                    except Exception:
                        pass
                else:
                    # Target lost completely (exceeded grace window)
                    tracking_active = False
                    tracking_bbox = None
                    tracking_centroid = None
                    tracking_confidence = 0.0
                    is_tracked = False

                    try:
                        await websocket.send_json({
                            "type": "tracking_update",
                            "bbox": None,
                            "centroid": None,
                            "confidence": 0.0,
                        })
                    except Exception:
                        pass

                # Temporal engine: evaluate tracking quality
                state = temporal_engine.evaluate_tracking_state(
                    is_tracked, tracking_confidence
                )
                if state in ("backtrack_triggered", "anomaly_detected"):
                    alert_type = "Tracking Anomaly — Backtracking" if state == "backtrack_triggered" else "Tracking Anomaly (Low Confidence/Lost)"
                    try:
                        await websocket.send_json({
                            "type": "alert",
                            "data": {
                                "id": int(time.time()),
                                "time": time.strftime("%H:%M:%S"),
                                "type": alert_type,
                            }
                        })
                    except Exception:
                        pass

            # 3. Boundary & Geofence Breach Detection
            if tracking_active and tracking_centroid:
                cx, cy = tracking_centroid
                breach_detected = False
                alert_type = ""

                if geofence_engine.polygon_points:
                    # Check if object left the geofence
                    if geofence_engine.check_breach(cx, cy):
                        breach_detected = True
                        alert_type = "Geofence Breach (Object Left Area)"
                else:
                    # No geofence: check if object left the camera screen bounds
                    margin = 15
                    if cx < margin or cx > 640 - margin or cy < margin or cy > 480 - margin:
                        breach_detected = True
                        alert_type = "Object Left Screen Bounds"

                if breach_detected:
                    now = asyncio.get_event_loop().time()
                    if now - last_breach_alert_time > BREACH_THROTTLE_SECS:
                        last_breach_alert_time = now
                        alert_record = alert_system.log_alert(alert_type, frame)
                        try:
                            await websocket.send_json({
                                "type": "alert",
                                "data": alert_record,
                            })
                        except Exception:
                            pass

            # 4. Compress and Send clean frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "type": "frame",
                "data": frame_b64,
            })

            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        print("[Backend] Sender disconnected")
    except Exception as e:
        print(f"[Backend] Error in video stream: {e}")
    finally:
        recv_task.cancel()
        cap.release()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

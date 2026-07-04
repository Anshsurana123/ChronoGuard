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

from tracker.privacy_filter import PrivacyFilter
from tracker.sam_client import SamClient
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

# ------------------------------------------------------------------
# Set your Colab ngrok URL here (or via SAM_ENDPOINT env var)
# ------------------------------------------------------------------
SAM_ENDPOINT = os.environ.get("SAM_ENDPOINT", "https://b2d9-34-6-88-71.ngrok-free.app")
sam_client = SamClient(endpoint_url=SAM_ENDPOINT)

temporal_engine = TemporalEngine(time_hop_interval=15)
geofence_engine = Geofence()
alert_system = AlertSystem()

# Shared state
current_frame = None
tracking_active = False
tracking_target = (320, 240)       # Initial click point
tracking_bbox = None               # [x, y, w, h] from SAM 3
tracking_centroid = None            # [cx, cy]  from SAM 3
tracking_confidence = 0.0
sam_connected = False               # Whether the Colab endpoint is reachable

@app.get("/")
async def root():
    return {"status": "ChronoGuard Local Backend Active", "sam_endpoint": SAM_ENDPOINT}

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    global current_frame, tracking_active
    global tracking_target, tracking_bbox, tracking_centroid, tracking_confidence
    global sam_connected

    await websocket.accept()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Background task to receive messages from Next.js
    async def receive_messages():
        global tracking_active, tracking_target
        global tracking_bbox, tracking_centroid, tracking_confidence
        global sam_connected
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "init_tracking":
                    print("[Backend] Received init_tracking:", msg)
                    click_x, click_y = msg["x"], msg["y"]
                    tracking_target = (click_x, click_y)
                    tracking_bbox = None
                    tracking_centroid = None
                    tracking_confidence = 0.0

                    if current_frame is not None:
                        # Call Colab SAM 3 endpoint
                        result = await sam_client.init_tracking(current_frame, click_x, click_y)
                        if result and result.get("bbox"):
                            tracking_active = True
                            tracking_bbox = result["bbox"]
                            tracking_centroid = result["centroid"]
                            tracking_confidence = result.get("confidence", 0)
                            sam_connected = True
                            print(f"[Backend] SAM 3 tracking initialized — bbox={tracking_bbox}, conf={tracking_confidence:.2f}")

                            # Send initial tracking data to frontend
                            await websocket.send_json({
                                "type": "tracking_update",
                                "bbox": tracking_bbox,
                                "centroid": tracking_centroid,
                                "confidence": tracking_confidence,
                            })
                        else:
                            # SAM endpoint unreachable — fall back to click-point only
                            tracking_active = True
                            tracking_centroid = [click_x, click_y]
                            sam_connected = False
                            print("[Backend] SAM 3 unreachable, tracking at click point only")

                            await websocket.send_json({
                                "type": "tracking_update",
                                "bbox": None,
                                "centroid": [click_x, click_y],
                                "confidence": 0,
                            })

                elif msg.get("type") == "set_geofence":
                    print("[Backend] Received geofence points:", msg["points"])
                    geofence_engine.set_polygon(msg["points"])

                elif msg.get("type") == "update_endpoint":
                    new_url = msg.get("endpoint_url", "").rstrip("/")
                    if new_url:
                        print(f"[Backend] Dynamic endpoint update request received: {new_url}")
                        await sam_client.update_endpoint(new_url)
                        sam_connected = False
                        await websocket.send_json({
                            "type": "endpoint_updated",
                            "endpoint_url": new_url,
                        })

                elif msg.get("type") == "stop_analysis":
                    tracking_active = False
                    tracking_bbox = None
                    tracking_centroid = None
                    tracking_confidence = 0.0
                    geofence_engine.set_polygon([])
                    await sam_client.reset()
                    print("[Backend] Analysis stopped, SAM session reset")

        except WebSocketDisconnect:
            print("[Backend] Receiver disconnected")

    recv_task = asyncio.create_task(receive_messages())

    last_breach_alert_time = 0
    BREACH_THROTTLE_SECS = 5
    frame_counter = 0
    # Only send frames to SAM every N video frames to avoid overloading
    SAM_UPDATE_INTERVAL = 3  # process every 3rd frame (~10 fps at 30fps capture)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame_counter += 1

            # Save raw frame for SAM init on click
            current_frame = frame.copy()

            # 1. Zero-Trust Privacy Filter
            frame = privacy_filter.apply(frame)

            # 2. SAM 3 Tracking — send frame to Colab for mask update
            if tracking_active and sam_connected and frame_counter % SAM_UPDATE_INTERVAL == 0:
                result = await sam_client.update_tracking(current_frame)
                if result and result.get("bbox"):
                    tracking_bbox = result["bbox"]
                    tracking_centroid = result["centroid"]
                    tracking_confidence = result.get("confidence", 0)

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

                    # Temporal engine: evaluate tracking quality
                    state = temporal_engine.evaluate_tracking_state(
                        True, tracking_confidence
                    )
                    if state == "backtrack_triggered":
                        try:
                            await websocket.send_json({
                                "type": "alert",
                                "data": {
                                    "id": int(asyncio.get_event_loop().time()),
                                    "time": __import__("time").strftime("%H:%M:%S"),
                                    "type": "Tracking Anomaly — Backtracking",
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
        await sam_client.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# ChronoGuard AI - Google Colab Endpoint Setup
# Copy and paste this code into a Google Colab Notebook cell

# ==========================================
# CELL 1: Install Dependencies
# ==========================================
import subprocess
import sys

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/sam2.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "pyngrok", "nest-asyncio", "python-multipart", "opencv-python-headless"])
    
    print("Downloading SAM 2.1 checkpoint...")
    subprocess.check_call("mkdir -p checkpoints", shell=True)
    subprocess.check_call("wget -q -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", shell=True)
    print("Setup complete.")

install_deps()

# ==========================================
# CELL 2: Start FastAPI Server with SAM 2 on Port 8080
# ==========================================
import os
import io
import cv2
import torch
import numpy as np
import base64
import tempfile
import shutil
from fastapi import FastAPI, Request
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from pydantic import BaseModel
from sam2.build_sam import build_sam2_video_predictor

# Set up device and optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # Enable TF32 for Ampere+ GPUs (like A100, RTX 3000+) to speed up SAM 2
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Load SAM 2.1 as a video predictor for true memory-based streaming
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

app = FastAPI()

# In-memory storage for active tracking sessions
# session_id -> {"state": inference_state, "tmp_dir": temp_path, "frame_count": int}
active_sessions = {}

class InitRequest(BaseModel):
    session_id: str
    frame_base64: str
    click_x: int
    click_y: int

class UpdateRequest(BaseModel):
    session_id: str
    frame_base64: str

def decode_image(b64_str):
    encoded_data = b64_str.split(',')[-1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def mask_to_bbox(mask_np):
    """Extract bounding box [x, y, w, h] from a binary mask."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def mask_to_centroid(mask_np):
    """Extract centroid (cx, cy) from a binary mask."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.mean()), int(ys.mean())]

def encode_mask(mask_np):
    """Encode a uint8 mask as base64 PNG."""
    _, buffer = cv2.imencode('.png', mask_np)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
async def root():
    return {"status": "ChronoGuard SAM 2 Endpoint is active and ready for inference!"}


@app.post("/init")
async def init_tracking(req: InitRequest):
    img = decode_image(req.frame_base64)
    # SAM 2 Video Predictor normally takes BGR as input if processing from JPEG directly via init_state,
    # but since it reads from disk it will load it as RGB internally. We just write the BGR image to disk.

    # Clean up any existing session
    if req.session_id in active_sessions:
        try:
            predictor.reset_state(active_sessions[req.session_id]["state"])
            shutil.rmtree(active_sessions[req.session_id]["tmp_dir"], ignore_errors=True)
        except Exception:
            pass

    # Write first frame to a temp dir — video predictor needs frames on disk
    tmp_dir = tempfile.mkdtemp()
    frame_path = os.path.join(tmp_dir, "00000.jpg")
    cv2.imwrite(frame_path, img)

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=tmp_dir)

        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[req.click_x, req.click_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    best_mask = (masks[0][0].cpu().numpy() > 0).astype(np.uint8)
    bbox = mask_to_bbox(best_mask)
    centroid = mask_to_centroid(best_mask)

    active_sessions[req.session_id] = {
        "state": inference_state,
        "tmp_dir": tmp_dir,
        "frame_count": 1,
    }

    return {
        "status": "initialized",
        "mask": encode_mask(best_mask * 255),
        "bbox": bbox,
        "centroid": centroid,
        "confidence": float(np.mean(best_mask)),
    }


MAX_FRAMES_ON_DISK = 7  # matches SAM 2's default num_maskmem

@app.post("/update")
async def update_tracking(req: UpdateRequest):
    if req.session_id not in active_sessions:
        return {"error": "Session not found. Call /init first."}

    session = active_sessions[req.session_id]
    inference_state = session["state"]
    tmp_dir = session["tmp_dir"]
    frame_idx = session["frame_count"]

    img = decode_image(req.frame_base64)

    # Write new frame to disk
    frame_path = os.path.join(tmp_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(frame_path, img)
    session["frame_count"] += 1

    best_mask = None
    bbox = None
    centroid = None

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        # propagate_in_video processes from start_frame_idx onward
        # It uses the memory bank built from all prior frames automatically
        for out_frame_idx, out_obj_ids, out_masks in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
        ):
            best_mask = (out_masks[0][0].cpu().numpy() > 0).astype(np.uint8)
            bbox = mask_to_bbox(best_mask)
            centroid = mask_to_centroid(best_mask)
            break  # we only want the current frame result

    # Prune old frames from disk to avoid filling Colab storage
    if frame_idx > MAX_FRAMES_ON_DISK:
        old_frame = os.path.join(tmp_dir, f"{frame_idx - MAX_FRAMES_ON_DISK:05d}.jpg")
        if os.path.exists(old_frame):
            os.remove(old_frame)

    if best_mask is None:
        return {"error": "Propagation returned no mask"}

    confidence = float(best_mask.mean()) if best_mask is not None else 0.0

    return {
        "status": "updated",
        "mask": encode_mask(best_mask * 255),
        "bbox": bbox,
        "centroid": centroid,
        "confidence": confidence,
    }


@app.post("/reset")
async def reset_session(req: dict):
    session_id = req.get("session_id", "")
    if session_id in active_sessions:
        session = active_sessions[session_id]
        try:
            predictor.reset_state(session["state"])
            shutil.rmtree(session["tmp_dir"], ignore_errors=True)
        except Exception:
            pass
        del active_sessions[session_id]
    return {"status": "session_reset"}


# ==========================================
# CELL 3: Start ngrok and Uvicorn
# ==========================================
PORT = 8000

try:
    from google.colab import userdata
    NGROK_AUTH_TOKEN = userdata.get('NGROK_TOKEN')
except Exception:
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"

if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    # Kill any existing ngrok tunnels just in case
    ngrok.kill()
    public_url = ngrok.connect(PORT)
    print(f"============================================")
    print(f"Public ngrok URL: {public_url.public_url}")
    print(f"============================================")
else:
    print("WARNING: Please set your NGROK_AUTH_TOKEN in Colab Secrets (Name: NGROK_TOKEN)")

if __name__ == "__main__":
    print("🚀 Starting FastAPI server... (This cell will keep running)")
    print("👀 Watch this output for incoming requests or errors.")
    
    # In Jupyter/Colab, an event loop is already running. 
    # Calling uvicorn.run() directly causes an asyncio error.
    # Instead, we create a config and await the server explicitly.
    import nest_asyncio
    nest_asyncio.apply()
    
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

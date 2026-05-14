# ChronoGuard AI - Google Colab Endpoint Setup (SAM 3)
# Copy and paste this code into a Google Colab Notebook cell

# ==========================================
# CELL 1: Install Dependencies
# ==========================================
import subprocess
import sys

def install_deps():
    print("Installing dependencies...")

    # ── SAM 3 core package ──────────────────────────────────────────────────
    # SAM 3 checkpoints are gated on HuggingFace (facebook/sam3).
    # You MUST have:
    #   1. Accepted the license at https://huggingface.co/facebook/sam3
    #   2. A HuggingFace access token added to Colab Secrets as "HF_TOKEN"
    # Once access is approved, this single pip install pulls everything.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/sam3.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "fastapi", "uvicorn", "pyngrok", "nest-asyncio",
                           "python-multipart", "opencv-python-headless",
                           "huggingface_hub"])

    print("Setup complete. Weights will be downloaded automatically via HuggingFace after login.")

install_deps()

# ==========================================
# CELL 2: Start FastAPI Server with SAM 3 on Port 8080
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

# ── SAM 3 imports ───────────────────────────────────────────────────────────
from sam3.model_builder import build_sam3_video_predictor

# ── HuggingFace authentication ──────────────────────────────────────────────
# SAM 3 weights are gated — authenticate before any model is built.
# Add your HuggingFace token to Colab Secrets (Left sidebar -> 🔑 Secrets)
# with the name "HF_TOKEN".
try:
    from google.colab import userdata
    from huggingface_hub import login as hf_login

    HF_TOKEN = userdata.get("HF_TOKEN")
    if HF_TOKEN:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        print("✅ HuggingFace authentication successful.")
    else:
        print("⚠️  WARNING: HF_TOKEN secret not set. SAM 3 weight download will fail.")
        print("   → Visit https://huggingface.co/facebook/sam3 and accept the license,")
        print("     then add your HF access token as 'HF_TOKEN' in Colab Secrets.")
except Exception as e:
    print(f"⚠️  HuggingFace auth skipped ({e}). Ensure you are authenticated separately.")

# ── Device setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # Enable TF32 for Ampere+ GPUs (A100, RTX 3000+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ── Build SAM 3 video predictor ──────────────────────────────────────────────
# SAM 3's builder resolves the checkpoint automatically via HuggingFace.
# No manual checkpoint path or YAML config is needed (unlike SAM 2).
gpus_to_use = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use if gpus_to_use else [0])

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
    return {"status": "ChronoGuard SAM 3 Endpoint is active and ready for inference!"}


@app.post("/init")
async def init_tracking(req: InitRequest):
    img = decode_image(req.frame_base64)

    # Clean up any existing session
    if req.session_id in active_sessions:
        try:
            old_session = active_sessions[req.session_id]
            if "real_session_id" in old_session:
                predictor.handle_request({
                    "type": "close_session",
                    "session_id": old_session["real_session_id"]
                })
            shutil.rmtree(old_session.get("tmp_dir", ""), ignore_errors=True)
        except Exception:
            pass

    # Write first frame to a temp dir — video predictor needs frames on disk
    tmp_dir = tempfile.mkdtemp()
    frame_path = os.path.join(tmp_dir, "00000.jpg")
    cv2.imwrite(frame_path, img)

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        # SAM 3: Use handle_request API
        session_res = predictor.handle_request({
            "type": "start_session",
            "resource_path": tmp_dir,
        })
        real_session_id = session_res["session_id"]

        prompt_res = predictor.handle_request({
            "type": "add_prompt",
            "session_id": real_session_id,
            "frame_index": 0,
            "obj_id": 1,
            "points": [[req.click_x, req.click_y]],
            "point_labels": [1],
        })

    # SAM 3's add_prompt returns:
    #   {"frame_index": int, "outputs": (obj_ids, low_res_masks, video_res_masks)}
    # where video_res_masks is a tensor of shape (num_objs, 1, H, W)
    outputs = prompt_res.get("outputs", None)

    best_mask = None
    if outputs is not None:
        if isinstance(outputs, tuple):
            # Expected format: (obj_ids, low_res_masks, video_res_masks)
            video_res_masks = outputs[-1]  # last element is video_res_masks
            if video_res_masks is not None and hasattr(video_res_masks, 'cpu'):
                best_mask = (video_res_masks[0, 0].cpu().numpy() > 0).astype(np.uint8)
        elif isinstance(outputs, dict):
            # Fallback: some SAM 3 versions may return a dict
            for key in ["video_res_masks", "out_binary_masks", "masks"]:
                if key in outputs:
                    m = outputs[key]
                    if hasattr(m, 'cpu'):
                        best_mask = (m[0, 0].cpu().numpy() > 0).astype(np.uint8)
                    else:
                        best_mask = (np.array(m)[0, 0] > 0).astype(np.uint8)
                    break
    
    if best_mask is None:
        best_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox = mask_to_bbox(best_mask)
    centroid = mask_to_centroid(best_mask)

    active_sessions[req.session_id] = {
        "state": None, # SAM 3 manages internal state via session_id
        "real_session_id": real_session_id,
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


MAX_FRAMES_ON_DISK = 7  # matches SAM 3's default memory window

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
        # SAM 3: propagate API using handle_stream_request
        for response in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session["real_session_id"],
            "start_frame_index": frame_idx,
            "propagation_direction": "forward",
            "max_frame_num_to_track": 1,
        }):
            outputs = response.get("outputs", None)
            if outputs is not None:
                if isinstance(outputs, tuple):
                    # Expected: (obj_ids, video_res_masks) or (obj_ids, low_res, video_res)
                    video_res_masks = outputs[-1]
                    if video_res_masks is not None and hasattr(video_res_masks, 'cpu'):
                        best_mask = (video_res_masks[0, 0].cpu().numpy() > 0).astype(np.uint8)
                        bbox = mask_to_bbox(best_mask)
                        centroid = mask_to_centroid(best_mask)
                elif isinstance(outputs, dict):
                    for key in ["video_res_masks", "out_binary_masks", "masks"]:
                        if key in outputs:
                            m = outputs[key]
                            if hasattr(m, 'cpu'):
                                best_mask = (m[0, 0].cpu().numpy() > 0).astype(np.uint8)
                            else:
                                best_mask = (np.array(m)[0, 0] > 0).astype(np.uint8)
                            bbox = mask_to_bbox(best_mask)
                            centroid = mask_to_centroid(best_mask)
                            break
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
            # SAM 3: close session
            predictor.handle_request({
                "type": "close_session",
                "session_id": session["real_session_id"]
            })
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
    print("🚀 Starting FastAPI server with SAM 3... (This cell will keep running)")
    print("👀 Watch this output for incoming requests or errors.")

    # In Jupyter/Colab, an event loop is already running.
    # Calling uvicorn.run() directly causes an asyncio error.
    # Instead, we create a config and await the server explicitly.
    import nest_asyncio
    nest_asyncio.apply()

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

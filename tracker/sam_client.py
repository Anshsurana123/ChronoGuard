import cv2
import base64
import json
import uuid
import asyncio
import aiohttp
import numpy as np


class SamClient:
    """
    Async HTTP client that talks to the SAM 3 inference endpoint
    running on Google Colab (exposed via ngrok).

    Returns structured tracking data: mask, bounding box, centroid,
    and confidence score.
    """

    def __init__(self, endpoint_url="http://localhost:8080"):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self._http_session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            # ngrok free tier serves an HTML interstitial page unless
            # this header is present — without it every request fails.
            headers = {"ngrok-skip-browser-warning": "true"}
            self._http_session = aiohttp.ClientSession(
                timeout=timeout, headers=headers
            )
        return self._http_session

    def _compress_and_encode(self, frame) -> str:
        """Compresses frame to reduce bandwidth for ngrok transmission."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64

    async def init_tracking(self, frame, click_x: int, click_y: int) -> dict | None:
        """
        Initializes the SAM 3 tracking session with the first frame
        and click coordinates.

        Returns dict with keys: mask, bbox, centroid, confidence
        or None on failure.
        """
        frame_b64 = self._compress_and_encode(frame)

        payload = {
            "session_id": self.session_id,
            "frame_base64": frame_b64,
            "click_x": int(click_x),
            "click_y": int(click_y),
        }

        try:
            session = await self._get_session()
            async with session.post(f"{self.endpoint_url}/init", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

            self.initialized = True
            return {
                "mask": data.get("mask"),
                "bbox": data.get("bbox"),        # [x, y, w, h]
                "centroid": data.get("centroid"),  # [cx, cy]
                "confidence": data.get("confidence", 0),
            }
        except Exception as e:
            print(f"[SamClient] Failed to initialize SAM 3 tracking: {e}")
            return None

    async def update_tracking(self, frame) -> dict | None:
        """
        Sends a subsequent frame to update the SAM 3 tracking state.

        Returns dict with keys: mask, bbox, centroid, confidence
        or None on failure.
        """
        if not self.initialized:
            print("[SamClient] Warning: SAM 3 not initialized. Call init_tracking first.")
            return None

        frame_b64 = self._compress_and_encode(frame)

        payload = {
            "session_id": self.session_id,
            "frame_base64": frame_b64,
        }

        try:
            session = await self._get_session()
            async with session.post(f"{self.endpoint_url}/update", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

            return {
                "mask": data.get("mask"),
                "bbox": data.get("bbox"),
                "centroid": data.get("centroid"),
                "confidence": data.get("confidence", 0),
            }
        except Exception as e:
            print(f"[SamClient] Failed to update SAM 3 tracking: {e}")
            return None

    async def reset(self):
        """Resets the current tracking session on the Colab endpoint."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.endpoint_url}/reset",
                json={"session_id": self.session_id},
            ) as resp:
                await resp.json()
        except Exception as e:
            print(f"[SamClient] Failed to reset session: {e}")

        # Generate a new session id for the next tracking run
        self.session_id = str(uuid.uuid4())
        self.initialized = False

    async def close(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

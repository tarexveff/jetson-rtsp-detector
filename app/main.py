"""
main.py – Flask web application entry point.

Starts the Detector background thread and serves:
  GET /          → viewer page (HTML)
  GET /stream    → MJPEG web preview stream
  GET /stats     → JSON stats (fps, detections, model, rtsp_output, …)
  GET /snapshot  → single latest JPEG frame
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator

from flask import Flask, Response, jsonify, render_template

from detector import Detector

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── config from environment ────────────────────────────────────────────────────
CAMERA_DEVICE  = os.environ.get("CAMERA_DEVICE",  "/dev/video0").strip()
CAMERA_WIDTH   = int(os.environ.get("CAMERA_WIDTH",  "1280"))
CAMERA_HEIGHT  = int(os.environ.get("CAMERA_HEIGHT", "720"))
CAMERA_FPS     = int(os.environ.get("CAMERA_FPS",    "30"))
CONFIDENCE     = float(os.environ.get("CONFIDENCE",  "0.40"))
WEB_PORT       = int(os.environ.get("WEB_PORT",      "8080"))
MODEL_SIZE     = os.environ.get("MODEL_SIZE",    "n").strip()
MODEL_DIR      = os.environ.get("MODEL_DIR",     "/models")
USE_TRT        = os.environ.get("USE_TENSORRT",  "0").strip() == "1"
RTSP_OUT_PORT  = int(os.environ.get("RTSP_OUT_PORT",  "8554"))
RTSP_OUT_PATH  = os.environ.get("RTSP_OUT_PATH", "/live").strip()
# UDP_DEST: IP of the machine that will play the stream.
# Use 127.0.0.1 to receive on the Jetson itself, or set to your viewer's IP.
UDP_DEST       = os.environ.get("UDP_DEST", "127.0.0.1").strip()

RTSP_OUTPUT_URL = f"udp://@:{RTSP_OUT_PORT}"

# ── detector ───────────────────────────────────────────────────────────────────
detector = Detector(
    camera_device=CAMERA_DEVICE,
    camera_width=CAMERA_WIDTH,
    camera_height=CAMERA_HEIGHT,
    camera_fps=CAMERA_FPS,
    model_size=MODEL_SIZE,
    confidence=CONFIDENCE,
    model_dir=MODEL_DIR,
    use_tensorrt=USE_TRT,
    rtsp_port=RTSP_OUT_PORT,
    rtsp_path=RTSP_OUT_PATH,
    udp_dest=UDP_DEST,
)
detector.start()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── placeholder frame (shown while the first real frame arrives) ───────────────
_LOADING_JPEG: bytes | None = None

def _loading_frame() -> bytes:
    global _LOADING_JPEG
    if _LOADING_JPEG is None:
        import numpy as np, cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Connecting to camera...", (60, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        _, buf = cv2.imencode(".jpg", img)
        _LOADING_JPEG = buf.tobytes()
    return _LOADING_JPEG


def _mjpeg_generator() -> Generator[bytes, None, None]:
    """Yield multipart MJPEG boundary chunks forever."""
    boundary = b"--frame"
    while True:
        frame = detector.latest_frame() or _loading_frame()
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
            b"\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps cap for the HTTP stream


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        camera_device=CAMERA_DEVICE,
        rtsp_output_url=RTSP_OUTPUT_URL,
        confidence=CONFIDENCE,
        model=f"YOLOv8{MODEL_SIZE}" + (" (TensorRT)" if USE_TRT else ""),
    )


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    return jsonify(detector.stats())


@app.route("/snapshot")
def snapshot():
    frame = detector.latest_frame() or _loading_frame()
    return Response(frame, mimetype="image/jpeg")


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(
        "Starting web server on 0.0.0.0:%d  |  camera: %s  |  UDP stream: %s",
        WEB_PORT, CAMERA_DEVICE, RTSP_OUTPUT_URL,
    )
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)

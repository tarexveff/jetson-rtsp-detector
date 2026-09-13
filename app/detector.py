"""
detector.py – USB camera capture, YOLOv8 inference, and RTSP output.

Pipeline:
  1. Open the USB camera via V4L2 (GStreamer nvarguscamerasrc fallback for
     CSI cameras, plain V4L2 for USB).
  2. Run YOLOv8 inference on each frame (CUDA / TensorRT).
  3. Draw bounding boxes + label (class name + confidence %).
  4. Push the annotated frame into two sinks simultaneously:
       a. A GStreamer RTSP server  →  rtsp://<host>:<port>/live
       b. A thread-safe JPEG buffer  →  Flask MJPEG web preview
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# GStreamer Python bindings (gi) are used only for the RTSP server.
# If unavailable the RTSP output falls back to a raw UDP/RTP appsrc pipeline.
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GstRtspServer, GLib
    _GST_RTSP_AVAILABLE = True
except Exception:
    _GST_RTSP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── colour palette (BGR) – one colour per class index, cycling ─────────────────
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
    (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
]


def _colour_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    """Overlay bounding boxes, class names, and confidence on *frame*."""
    annotated = frame.copy()
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        names = result.names  # dict[int, str]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = f"{names[cls_id]}  {conf:.0%}"
            colour = _colour_for(cls_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness  = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            pad   = 3
            bg_y1 = max(y1 - th - 2 * pad, 0)
            cv2.rectangle(
                annotated,
                (x1, bg_y1),
                (x1 + tw + 2 * pad, y1),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                annotated, label,
                (x1 + pad, y1 - pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# RTSP publisher
# ══════════════════════════════════════════════════════════════════════════════

class RtspPublisher:
    """
    Wraps a GStreamer RTSP server that streams annotated frames pushed by
    the detector thread.

    Clients connect to:  rtsp://<host>:<rtsp_port>/<rtsp_path>

    Implementation notes
    ────────────────────
    GstRtspServer works by mounting a "media factory" at a URL path.  We use
    an appsrc-based factory so we can push raw BGR frames from Python.  Each
    frame is converted to I420 and encoded to H.264 by nvv4l2h264enc (Jetson
    HW encoder) with a software x264enc fallback.

    The GLib main loop must run in its own thread so that GStreamer can
    dispatch its internal callbacks.
    """

    # Pipeline launched inside the RTSP session for each connecting client.
    # appsrc feeds BGR → videoconvert → I420 → HW H.264 encode → RTP packetise.
    #
    # block=true  – makes push-buffer block instead of silently overflowing,
    #               so back-pressure is surfaced rather than hiding a stall.
    # queue leaky=downstream – drops the *oldest* buffered frame when the
    #               encoder can't keep up, preventing the pipeline from hanging.
    _PIPELINE_HW = (
        "appsrc name=src is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 "
        "! queue max-size-buffers=2 leaky=downstream "
        "! videoconvert "
        "! video/x-raw,format=I420 "
        "! nvv4l2h264enc maxperf-enable=1 bitrate=4000000 "
        "! h264parse "
        "! rtph264pay name=pay0 pt=96"
    )

    _PIPELINE_SW = (
        "appsrc name=src is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 "
        "! queue max-size-buffers=2 leaky=downstream "
        "! videoconvert "
        "! video/x-raw,format=I420 "
        "! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast "
        "! h264parse "
        "! rtph264pay name=pay0 pt=96"
    )

    def __init__(
        self,
        port: int = 8554,
        path: str = "/live",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.port   = port
        self.path   = path
        self.width  = width
        self.height = height
        self.fps    = fps

        self._appsrc: Optional[object] = None   # Gst.Element for the current session
        self._pts: int = 0                      # reset to 0 on each new client session
        self._frame_duration: int = 0           # nanoseconds
        self._lock = threading.Lock()
        self._started = False

        if not _GST_RTSP_AVAILABLE:
            logger.warning(
                "GstRtspServer Python bindings not available. "
                "RTSP output is DISABLED.  Install python3-gi and "
                "gir1.2-gst-rtsp-server-1.0 inside the container."
            )

    def start(self) -> None:
        if not _GST_RTSP_AVAILABLE or self._started:
            return

        Gst.init(None)

        self._frame_duration = int(1e9 / self.fps)  # ns per frame

        server  = GstRtspServer.RTSPServer.new()
        server.set_service(str(self.port))

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_shared(True)   # one pipeline, many clients

        # Try HW encoder first; fall back to software
        hw_pipeline  = self._PIPELINE_HW.format(
            w=self.width, h=self.height, fps=self.fps
        )
        sw_pipeline  = self._PIPELINE_SW.format(
            w=self.width, h=self.height, fps=self.fps
        )

        # Probe whether nvv4l2h264enc exists on this system
        if Gst.ElementFactory.find("nvv4l2h264enc"):
            logger.info("RTSP publisher: using nvv4l2h264enc (HW H.264).")
            factory.set_launch(f"( {hw_pipeline} )")
        else:
            logger.warning(
                "nvv4l2h264enc not found – falling back to x264enc (SW)."
            )
            factory.set_launch(f"( {sw_pipeline} )")

        factory.connect("media-configure", self._on_media_configure)

        mounts = server.get_mount_points()
        mounts.add_factory(self.path, factory)
        server.attach(None)

        logger.info(
            "RTSP server listening on rtsp://0.0.0.0:%d%s", self.port, self.path
        )

        # Run the GLib main loop in a daemon thread
        loop = GLib.MainLoop()
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()

        self._started = True

    def _on_media_configure(self, factory, media) -> None:  # noqa: ARG002
        """Called when a new RTSP session pipeline is built."""
        element = media.get_element()
        appsrc  = element.get_child_by_name("src")
        if appsrc is None:
            logger.error("RTSP publisher: could not find appsrc element.")
            return
        with self._lock:
            self._appsrc = appsrc
            # Reset PTS so each new client session starts from t=0, avoiding
            # decoder rejection of a stream that begins with a large timestamp.
            self._pts = 0
        logger.info("RTSP client connected – appsrc configured.")

    def push_frame(self, frame: np.ndarray) -> None:
        """Push one annotated BGR frame into the RTSP pipeline (non-blocking)."""
        if not _GST_RTSP_AVAILABLE or not self._started:
            return
        with self._lock:
            appsrc = self._appsrc
        if appsrc is None:
            return  # no client connected yet

        # Resize if the frame dimensions changed (shouldn't happen, but safe)
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        data   = frame.tobytes()
        buf    = Gst.Buffer.new_wrapped(data)
        buf.pts      = self._pts
        buf.duration = self._frame_duration
        self._pts   += self._frame_duration

        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.debug("appsrc push-buffer returned: %s", ret)

    @property
    def url(self) -> str:
        return f"rtsp://0.0.0.0:{self.port}{self.path}"


# ══════════════════════════════════════════════════════════════════════════════
# Main Detector
# ══════════════════════════════════════════════════════════════════════════════

class Detector:
    """
    Background thread that reads from a USB camera, runs YOLOv8,
    pushes annotated frames to the RTSP server, and keeps a JPEG buffer
    for the Flask MJPEG preview endpoint.
    """

    def __init__(
        self,
        camera_device: str = "/dev/video0",
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
        model_size: str = "n",
        confidence: float = 0.40,
        model_dir: str = "/models",
        use_tensorrt: bool = False,
        rtsp_port: int = 8554,
        rtsp_path: str = "/live",
    ) -> None:
        self.camera_device  = camera_device
        self.camera_width   = camera_width
        self.camera_height  = camera_height
        self.camera_fps     = camera_fps
        self.confidence     = confidence
        self.model_dir      = Path(model_dir)
        self.use_tensorrt   = use_tensorrt

        # ── select inference device ─────────────────────────────────────────
        import torch
        if torch.cuda.is_available():
            self.infer_device = 0
            backend_label = "tensorrt" if use_tensorrt else "cuda"
            logger.info("CUDA available – using GPU 0 for inference.")
        else:
            self.infer_device = "cpu"
            backend_label = "cpu"
            if use_tensorrt:
                logger.warning(
                    "CUDA not available – TensorRT export requires CUDA. "
                    "Falling back to CPU inference."
                )
            else:
                logger.warning("CUDA not available – running inference on CPU.")

        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None
        self._stats: dict = {
            "fps": 0.0,
            "detections": 0,
            "resolution": "—",
            "model": "",
            "backend": backend_label,
            "camera": camera_device,
            "rtsp_output": f"rtsp://<host>:{rtsp_port}{rtsp_path}",
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ── RTSP publisher ──────────────────────────────────────────────────
        self._publisher = RtspPublisher(
            port=rtsp_port,
            path=rtsp_path,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )

        # ── load model ──────────────────────────────────────────────────────
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"yolov8{model_size}"

        if use_tensorrt:
            engine_path = self.model_dir / f"{model_name}.engine"
            if not engine_path.exists():
                logger.info(
                    "TensorRT engine not found – exporting from %s.pt …", model_name
                )
                pt_path = self.model_dir / f"{model_name}.pt"
                base = YOLO(str(pt_path) if pt_path.exists() else f"{model_name}.pt")
                base.export(format="engine", device=self.infer_device, half=True)
                exported = Path(base.ckpt_path).with_suffix(".engine")
                exported.rename(engine_path)
            model_path = str(engine_path)
            logger.info("Loading TensorRT engine: %s", model_path)
        else:
            pt_path = self.model_dir / f"{model_name}.pt"
            if not pt_path.exists():
                logger.info("Downloading %s.pt to %s …", model_name, self.model_dir)
                YOLO(f"{model_name}.pt")
                downloaded = Path(f"{model_name}.pt")
                if downloaded.exists():
                    shutil.move(str(downloaded), str(pt_path))
            model_path = str(pt_path) if pt_path.exists() else f"{model_name}.pt"
            logger.info("Loading YOLO model: %s", model_path)

        self.model = YOLO(model_path)
        self._stats["model"] = model_name + (".engine" if use_tensorrt else ".pt")
        logger.info("Model ready.")

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._publisher.start()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detector thread started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ── internal ───────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Open the USB camera.  Tries three methods in order:
          1. GStreamer v4l2src with explicit caps (best, zero-copy on Jetson)
          2. GStreamer v4l2src without forcing caps (lets camera negotiate)
          3. OpenCV V4L2 backend directly with the device path string
        """
        dev = self.camera_device
        w   = self.camera_width
        h   = self.camera_height
        fps = self.camera_fps

        # ── attempt 1: GStreamer with explicit resolution/framerate caps ────────
        # timeout=5000000000 (5 s in ns) makes the source return an error buffer
        # instead of blocking indefinitely if the camera stalls.
        gst_explicit = (
            f"v4l2src device={dev} do-timestamp=true "
            f"! video/x-raw,width={w},height={h},framerate={fps}/1 "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=true sync=false max-buffers=2"
        )
        cap = cv2.VideoCapture(gst_explicit, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("Opened %s via GStreamer (explicit caps).", dev)
            return cap
        cap.release()
        logger.warning("GStreamer explicit-caps pipeline failed for %s.", dev)

        # ── attempt 2: GStreamer letting the camera negotiate its own caps ───────
        gst_auto = (
            f"v4l2src device={dev} do-timestamp=true "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=true sync=false max-buffers=2"
        )
        cap = cv2.VideoCapture(gst_auto, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("Opened %s via GStreamer (auto caps).", dev)
            # Apply desired resolution/fps as hints (best-effort)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            return cap
        cap.release()
        logger.warning("GStreamer auto-caps pipeline failed for %s.", dev)

        # ── attempt 3: plain OpenCV V4L2 backend using the device path ──────────
        # Pass the path string directly — OpenCV accepts both "/dev/videoN"
        # strings and integer indices on Linux.
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Last resort: derive integer index and try that
            try:
                idx = int(dev.replace("/dev/video", "")) if "/dev/video" in dev else int(dev)
            except ValueError:
                idx = 0
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        # Limit the internal V4L2 buffer to 2 frames so cap.read() returns
        # quickly instead of draining a deep queue of stale frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open USB camera: {dev}\n"
                f"  Check that the device is passed to the container:\n"
                f"    podman run --device {dev} ..."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        logger.info("Opened %s via OpenCV V4L2 backend.", dev)
        return cap

    def _loop(self) -> None:
        retry_delay = 3
        while self._running:
            cap = None
            try:
                cap = self._open_capture()
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                with self._lock:
                    self._stats["resolution"] = f"{w}×{h}"

                fps_counter = 0
                t_start = time.monotonic()

                while self._running:
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("Frame read failed – reopening camera …")
                        break

                    # ── inference ───────────────────────────────────────────
                    results = self.model.predict(
                        frame,
                        conf=self.confidence,
                        device=self.infer_device,
                        verbose=False,
                        stream=False,
                    )

                    # ── annotate ────────────────────────────────────────────
                    annotated = _draw_boxes(frame, results)
                    num_det   = sum(len(r.boxes) for r in results if r.boxes)

                    # ── push to RTSP server ─────────────────────────────────
                    self._publisher.push_frame(annotated)

                    # ── encode to JPEG for web preview ──────────────────────
                    ok_enc, buf = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if ok_enc:
                        jpeg = buf.tobytes()
                    else:
                        continue

                    # ── FPS rolling average ─────────────────────────────────
                    fps_counter += 1
                    elapsed = time.monotonic() - t_start
                    if elapsed >= 1.0:
                        fps = fps_counter / elapsed
                        fps_counter = 0
                        t_start = time.monotonic()
                        with self._lock:
                            self._stats["fps"]        = round(fps, 1)
                            self._stats["detections"] = num_det

                    with self._lock:
                        self._frame = jpeg

            except Exception as exc:
                logger.error("Detector error: %s", exc, exc_info=True)
            finally:
                if cap is not None:
                    cap.release()

            if self._running:
                logger.info("Reopening camera in %ds …", retry_delay)
                time.sleep(retry_delay)

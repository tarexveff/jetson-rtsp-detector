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


# gi/GStreamer is used directly for the UDP output pipeline since OpenCV
# in the dustynv base image is built without GStreamer write support.
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GLib
    _GST_AVAILABLE = True
except Exception:
    _GST_AVAILABLE = False

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
# RTP/UDP publisher — uses gi/GStreamer directly (OpenCV has no GStreamer write)
# ══════════════════════════════════════════════════════════════════════════════

class RtspPublisher:
    """
    Encodes annotated frames as H.264 and sends them as RTP over UDP using
    a GStreamer pipeline driven directly via Python gi bindings.

    Pipeline:
      appsrc → videoconvert → x264enc → h264parse → rtph264pay → udpsink

    OpenCV in the dustynv/pytorch image is built WITHOUT GStreamer write
    support, so cv2.VideoWriter cannot be used here.  Instead we build the
    pipeline with Gst.parse_launch, grab the appsrc element, and push raw
    BGR frames as GstBuffers from push_frame().

    Clients play with:
      ffplay udp://@:<port>?overrun_nonfatal=1&fifo_size=50000000
      vlc    udp://@:<port>
    """

    def __init__(
        self,
        port: int = 8554,
        path: str = "/live",   # kept for API compat
        dest_host: str = "127.0.0.1",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.port      = port
        self.dest_host = dest_host
        self.width     = width
        self.height    = height
        self.fps       = fps

        self._pipeline   = None
        self._appsrc     = None
        self._pts: int   = 0
        self._frame_dur  = int(1e9 / fps)   # nanoseconds
        self._lock       = threading.Lock()
        self._started    = False

    def start(self) -> None:
        if not _GST_AVAILABLE:
            logger.error(
                "GStreamer Python bindings (gi) not available – "
                "UDP stream disabled.  Web preview still works."
            )
            return

        Gst.init(None)

        for label, desc in (
            ("nvv4l2h264enc (HW)", self._hw_desc()),
            ("x264enc (SW)",       self._sw_desc()),
        ):
            logger.info("Trying UDP pipeline: %s", label)
            try:
                pipeline = Gst.parse_launch(desc)
            except Exception as exc:
                logger.warning("Pipeline parse failed [%s]: %s", label, exc)
                continue

            appsrc = pipeline.get_by_name("src")
            if appsrc is None:
                logger.warning("No appsrc in pipeline [%s]", label)
                continue

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                logger.warning("Pipeline failed to start [%s]", label)
                continue

            self._pipeline  = pipeline
            self._appsrc    = appsrc
            self._started   = True

            # GLib main loop — needed for GStreamer bus messages
            loop = GLib.MainLoop()
            threading.Thread(target=loop.run, daemon=True).start()

            logger.info(
                "UDP stream running [%s] → udp://@:%d  "
                "  Play: ffplay udp://@:%d  |  vlc udp://@:%d",
                label, self.port, self.port, self.port,
            )
            return

        logger.error(
            "All UDP pipelines failed. "
            "Check: gst-inspect-1.0 x264enc  and  gst-inspect-1.0 udpsink. "
            "Web preview is still available."
        )

    def _hw_desc(self) -> str:
        return (
            f"appsrc name=src is-live=true block=false format=time "
            f"max-bytes=0 max-buffers=2 leaky-type=downstream "
            f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 "
            f"! videoconvert ! video/x-raw,format=I420 "
            f"! nvv4l2h264enc maxperf-enable=1 bitrate=4000000 iframeinterval=30 "
            f"! h264parse config-interval=-1 "
            f"! rtph264pay pt=96 config-interval=-1 "
            f"! udpsink host={self.dest_host} port={self.port} sync=false"
        )

    def _sw_desc(self) -> str:
        return (
            f"appsrc name=src is-live=true block=false format=time "
            f"max-bytes=0 max-buffers=2 leaky-type=downstream "
            f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 "
            f"! videoconvert ! video/x-raw,format=I420 "
            f"! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=30 "
            f"! h264parse config-interval=-1 "
            f"! rtph264pay pt=96 config-interval=-1 "
            f"! udpsink host={self.dest_host} port={self.port} sync=false"
        )

    def push_frame(self, frame: np.ndarray) -> None:
        if not self._started or self._appsrc is None:
            return

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        buf = Gst.Buffer.new_wrapped(frame.tobytes())
        buf.pts      = self._pts
        buf.dts      = self._pts
        buf.duration = self._frame_dur
        self._pts   += self._frame_dur

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.warning("appsrc push-buffer returned %s — resetting PTS", ret)
            with self._lock:
                self._pts = 0

    @property
    def url(self) -> str:
        return f"udp://@:{self.port}"


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
        udp_dest: str = "127.0.0.1",
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
            "rtsp_output": f"udp://@:{rtsp_port}",
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ── UDP publisher ───────────────────────────────────────────────────
        self._publisher = RtspPublisher(
            port=rtsp_port,
            path=rtsp_path,
            dest_host=udp_dest,
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
        Open the USB camera via OpenCV V4L2 backend.
        GStreamer read is skipped — this OpenCV build has GStreamer: NO,
        so CAP_V4L2 is the only working path.
        """
        dev = self.camera_device
        w   = self.camera_width
        h   = self.camera_height
        fps = self.camera_fps

        # Try the device path string directly first (works on Linux)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Derive integer index (/dev/video0 → 0) as fallback
            try:
                idx = int(dev.replace("/dev/video", "")) if "/dev/video" in dev else int(dev)
            except ValueError:
                idx = 0
            logger.debug("Device path failed, trying index %d", idx)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open USB camera: {dev}\n"
                f"  Check that the device is passed to the container:\n"
                f"    podman run --device {dev} ..."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Opened %s via V4L2 (%dx%d).", dev, actual_w, actual_h)
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

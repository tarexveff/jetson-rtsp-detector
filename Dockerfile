# ──────────────────────────────────────────────────────────────────────────────
# Base: dustynv/pytorch — public Jetson containers, no NGC account needed.
# https://github.com/dusty-nv/jetson-containers
#
# Pick the tag that matches your JetPack  (cat /etc/nv_tegra_release):
#
#   R36.5.0 / JetPack 6.2 (CUDA 12.8, Ubuntu 24.04)
#     → 2.7-r36.4.0-cu128-24.04                        ← default (current)
#
#   R36.4.0 / JetPack 6.1 (CUDA 12.8, Ubuntu 24.04)
#     → 2.7-r36.4.0-cu128-24.04
#
#   R36.2.0 / JetPack 6.0 (CUDA 12.2, Ubuntu 22.04)
#     → r36.2.0-cu122-torch2.2-ubuntu22.04
#
#   R35.4.1 / JetPack 5.1.3 (CUDA 11.4, Ubuntu 20.04)
#     → r35.4.1-cu114-torch2.1-ubuntu20.04
#
# To list all available tags for your revision:
#   curl -s "https://hub.docker.com/v2/repositories/dustynv/pytorch/tags?page_size=100" \
#     | python3 -c "import json,sys; [print(t['name']) for t in json.load(sys.stdin)['results'] if 'r36' in t['name']]"
#
# Full tag list: https://hub.docker.com/r/dustynv/pytorch/tags
# ──────────────────────────────────────────────────────────────────────────────
ARG JETPACK_TAG=2.7-r36.4.0-cu128-24.04
FROM dustynv/pytorch:${JETPACK_TAG}

LABEL maintainer="jetson-rtsp-detector"
LABEL description="USB camera → YOLOv8 object detection → RTSP output on Jetson Orin Nano"

# ── system packages ────────────────────────────────────────────────────────────
# Note: no inline # comments inside RUN blocks — they break apt on Ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        libglib2.0-0 \
        libxext6 \
        libgl1 \
        libglx-mesa0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-rtsp \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstrtspserver-1.0-0 \
        libgstrtspserver-1.0-dev \
        gir1.2-gst-rtsp-server-1.0 \
        python3-gi \
        python3-gi-cairo \
        gir1.2-glib-2.0 \
        v4l-utils \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# The dustynv base image uses a venv at /opt/venv — pip3 points into it.
# Do NOT install opencv-python: the base image has a GStreamer+CUDA build
# of OpenCV already; a pip wheel would overwrite it with a CPU-only version.
# Reset the index to PyPI (the Jetson index lacks general packages like flask).
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir \
        --index-url https://pypi.org/simple/ \
        --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu128 \
        -r /tmp/requirements.txt

# ── application code ───────────────────────────────────────────────────────────
WORKDIR /app

# Model weights are downloaded on first run and cached in /models.
# Mount a host volume here to avoid re-downloading across container restarts:
#   -v /path/on/host/models:/models
RUN mkdir -p /models
ENV MODEL_DIR=/models

# ── runtime defaults (all overridable with -e at podman run) ───────────────────
ENV CAMERA_DEVICE=/dev/video0
ENV CAMERA_WIDTH=640
ENV CAMERA_HEIGHT=480
ENV CAMERA_FPS=30
ENV CONFIDENCE=0.40
ENV WEB_PORT=8080
ENV MODEL_SIZE=n
# Set to 1 to export/use a TensorRT engine (slower first start, faster inference)
ENV USE_TENSORRT=0
ENV RTSP_OUT_PORT=8554
ENV RTSP_OUT_PATH=/live

# Web preview port
EXPOSE 8080
# RTSP output port
EXPOSE 8554

CMD ["python3", "main.py"]

COPY app/ /app/
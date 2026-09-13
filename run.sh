#!/usr/bin/env bash
# run.sh – build and launch the USB-camera object detector on a Jetson Orin Nano
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                          # uses /dev/video0
#   ./run.sh /dev/video2              # specific USB camera device
#
# Optional environment variables (export before running, or edit defaults below):
#   CAMERA_DEVICE   – V4L2 device node            (default: /dev/video0)
#   CAMERA_WIDTH    – capture width  in pixels     (default: 1280)
#   CAMERA_HEIGHT   – capture height in pixels     (default: 720)
#   CAMERA_FPS      – capture frame rate           (default: 30)
#   CONFIDENCE      – detection confidence 0–1     (default: 0.40)
#   MODEL_SIZE      – n | s | m | l | x            (default: n)
#   WEB_PORT        – host port for the web UI     (default: 8080)
#   RTSP_OUT_PORT   – RTSP server port             (default: 8554)
#   RTSP_OUT_PATH   – RTSP mount path              (default: /live)
#   USE_TENSORRT    – 1 to enable TRT export       (default: 0)
#   IMAGE_NAME      – container image tag          (default: rtsp-detector)
#   MODEL_CACHE_DIR – host path for model weights  (default: ./models)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
# JETPACK_TAG selects the dustynv/pytorch base image.
# Run `cat /etc/nv_tegra_release` on the Jetson to find your JetPack version.
#   JetPack 6.0   → r36.2.0-cu122-torch2.2-ubuntu22.04  (default)
#   JetPack 5.1.2 → r35.3.1-cu114-torch2.1-ubuntu20.04
#   JetPack 5.1.1 → r35.2.1-cu114-torch2.0-ubuntu20.04
JETPACK_TAG="${JETPACK_TAG:-2.7-r36.4.0-cu128-24.04}"
CAMERA_DEVICE="${CAMERA_DEVICE:-${1:-/dev/video0}}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CONFIDENCE="${CONFIDENCE:-0.40}"
MODEL_SIZE="${MODEL_SIZE:-n}"
WEB_PORT="${WEB_PORT:-8080}"
RTSP_OUT_PORT="${RTSP_OUT_PORT:-8554}"
RTSP_OUT_PATH="${RTSP_OUT_PATH:-/live}"
# UDP_DEST: IP address of the machine that will play the stream.
# Change this to the IP of your viewing machine, e.g. 192.168.1.50
UDP_DEST="${UDP_DEST:-127.0.0.1}"
USE_TENSORRT="${USE_TENSORRT:-0}"
IMAGE_NAME="${IMAGE_NAME:-rtsp-detector}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/models}"
CONTAINER_NAME="rtsp-detector"

# ── validate camera device ────────────────────────────────────────────────────
if [[ ! -e "${CAMERA_DEVICE}" ]]; then
  echo "WARNING: ${CAMERA_DEVICE} not found on this host."
  echo "  Available video devices:"
  ls /dev/video* 2>/dev/null || echo "  (none)"
  echo "  Continuing – the device will be checked again at container start."
fi

mkdir -p "${MODEL_CACHE_DIR}"

HOST_IP=$(hostname -I | awk '{print $1}')

# ── build ─────────────────────────────────────────────────────────────────────
echo "► Building image: ${IMAGE_NAME} …"
echo "   JetPack tag : ${JETPACK_TAG}"
podman build \
  --tag "${IMAGE_NAME}:latest" \
  --file Dockerfile \
  --build-arg JETPACK_TAG="${JETPACK_TAG}" \
  --network=host \
  .

# ── resolve video group ID so the container can access /dev/videoN ───────────
VIDEO_GID=$(getent group video 2>/dev/null | cut -d: -f3 || stat -c '%g' "${CAMERA_DEVICE}" 2>/dev/null || echo "")

# ── mount host Jetson CUDA/tegra libraries ───────────────────────────────────
# Strategy: mount the tegra dir to a NEUTRAL path (/host-tegra) so it never
# shadows the container's own /usr/lib/aarch64-linux-gnu/gstreamer-1.0 dir.
# Then set LD_LIBRARY_PATH to include /host-tegra so PyTorch finds libcuda.so.
# The Jetson GStreamer NV plugins are mounted to a separate subdir and added
# to GST_PLUGIN_PATH so nvv4l2h264enc etc. load from the real host .so files.
CUDA_MOUNTS=()

# Tegra runtime libs → neutral mount point (no collision with container paths)
if [[ -d /usr/lib/aarch64-linux-gnu/tegra ]]; then
  CUDA_MOUNTS+=("--volume" "/usr/lib/aarch64-linux-gnu/tegra:/host-tegra:ro")
  echo "   tegra libs  : mounted → /host-tegra"
fi
if [[ -d /usr/lib/aarch64-linux-gnu/tegra-egl ]]; then
  CUDA_MOUNTS+=("--volume" "/usr/lib/aarch64-linux-gnu/tegra-egl:/host-tegra-egl:ro")
fi

# CUDA toolkit dirs (safe to mount at their original paths — no plugin clash)
for lib_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/lib64/nvidia; do
  [[ -d "$lib_path" ]] && CUDA_MOUNTS+=("--volume" "${lib_path}:${lib_path}:ro")
done

echo "   CUDA mounts : ${#CUDA_MOUNTS[@]} total"

# ── stop any existing instance ────────────────────────────────────────────────
podman rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── run ───────────────────────────────────────────────────────────────────────
echo "► Starting container '${CONTAINER_NAME}' …"
echo "   Camera      : ${CAMERA_DEVICE}  (${CAMERA_WIDTH}×${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps)"
echo "   Model       : YOLOv8${MODEL_SIZE}  (TensorRT: ${USE_TENSORRT})"
echo "   Confidence  : ${CONFIDENCE}"
echo "   RTSP output : rtsp://${HOST_IP}:${RTSP_OUT_PORT}${RTSP_OUT_PATH}"
echo "   Web preview : http://${HOST_IP}:${WEB_PORT}"

# Build optional group-add flag only if we found the video GID
GROUP_ADD=()
[[ -n "${VIDEO_GID}" ]] && GROUP_ADD+=("--group-add" "${VIDEO_GID}")

podman run \
  --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  \
  --privileged \
  --user root \
  "${GROUP_ADD[@]}" \
  \
  --device "${CAMERA_DEVICE}" \
  \
  --network host \
  --shm-size 512m \
  \
  "${CUDA_MOUNTS[@]}" \
  --volume "${MODEL_CACHE_DIR}:/models:z" \
  \
  --env CAMERA_DEVICE="${CAMERA_DEVICE}" \
  --env CAMERA_WIDTH="${CAMERA_WIDTH}" \
  --env CAMERA_HEIGHT="${CAMERA_HEIGHT}" \
  --env CAMERA_FPS="${CAMERA_FPS}" \
  --env CONFIDENCE="${CONFIDENCE}" \
  --env MODEL_SIZE="${MODEL_SIZE}" \
  --env WEB_PORT="${WEB_PORT}" \
  --env RTSP_OUT_PORT="${RTSP_OUT_PORT}" \
  --env RTSP_OUT_PATH="${RTSP_OUT_PATH}" \
  --env USE_TENSORRT="${USE_TENSORRT}" \
  --env UDP_DEST="${UDP_DEST}" \
  --env LD_LIBRARY_PATH="/host-tegra:/host-tegra-egl:/usr/lib64/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
  \
  "${IMAGE_NAME}:latest"

echo ""
echo "✓ Container started."
echo ""
echo "  UDP stream  : udp://@:${RTSP_OUT_PORT}"
echo "              Play with:  ffplay \"udp://@:${RTSP_OUT_PORT}?overrun_nonfatal=1&fifo_size=50000000\""
echo "              Or:         vlc    udp://@:${RTSP_OUT_PORT}"
echo ""
echo "  Web preview : http://${HOST_IP}:${WEB_PORT}"
echo ""
echo "  Logs:    podman logs -f ${CONTAINER_NAME}"
echo "  Stop:    podman stop ${CONTAINER_NAME}"
echo "  Remove:  podman rm ${CONTAINER_NAME}"

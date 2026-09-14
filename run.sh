#!/usr/bin/env bash
# run.sh – build and launch the object detector on a Jetson Orin Nano
#
# CAMERA_DEVICE can be a local V4L2 device OR a URL:
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                          # uses /dev/video0 (default)
#   ./run.sh /dev/video2              # specific local device
#   CAMERA_DEVICE=rtsp://user:pass@192.168.1.50/stream1 ./run.sh   # RTSP URL
#   CAMERA_DEVICE=http://192.168.1.60/video ./run.sh                # HTTP/MJPEG
#
# Optional environment variables (export before running, or edit defaults below):
#   CAMERA_DEVICE   – V4L2 device node (e.g. /dev/video0) OR a URL
#                     (rtsp://, rtsps://, rtmp://, http://, https://)
#                     (default: /dev/video0)
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
USE_TENSORRT="${USE_TENSORRT:-0}"
IMAGE_NAME="${IMAGE_NAME:-rtsp-detector}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/models}"
CONTAINER_NAME="rtsp-detector"

# ── validate camera device (skip check for URLs) ─────────────────────────────
if [[ "${CAMERA_DEVICE}" != rtsp://* ]] && \
   [[ "${CAMERA_DEVICE}" != rtsps://* ]] && \
   [[ "${CAMERA_DEVICE}" != rtmp://* ]] && \
   [[ "${CAMERA_DEVICE}" != http://* ]] && \
   [[ "${CAMERA_DEVICE}" != https://* ]] && \
   [[ ! -e "${CAMERA_DEVICE}" ]]; then
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
# Only relevant for local devices; URLs don't require device permissions.
VIDEO_GID=$(getent group video 2>/dev/null | cut -d: -f3 || stat -c '%g' "${CAMERA_DEVICE}" 2>/dev/null || echo "")

# ── mount host Jetson CUDA/tegra userspace libraries ─────────────────────────
# On Jetson the CUDA runtime is NOT inside the container image — it lives on
# the host BSP.  We bind-mount the relevant host paths so the container's
# PyTorch/CUDA can find libcuda.so, libnvrtc.so, etc.
# Paths that don't exist on this host are simply skipped.
CUDA_MOUNTS=()
for lib_path in \
    /usr/lib/aarch64-linux-gnu/tegra \
    /usr/lib/aarch64-linux-gnu/tegra-egl \
    /usr/lib64/nvidia \
    /usr/local/cuda \
    /usr/local/cuda-12 \
    /usr/local/cuda-12.8 \
; do
  [[ -e "$lib_path" ]] && CUDA_MOUNTS+=("--volume" "${lib_path}:${lib_path}:ro")
done

# libcuda.so is sometimes a standalone file outside the tegra dir
for lib_file in \
    /usr/lib/aarch64-linux-gnu/libcuda.so.1 \
    /usr/lib/aarch64-linux-gnu/libcuda.so \
; do
  [[ -e "$lib_file" ]] && CUDA_MOUNTS+=("--volume" "${lib_file}:${lib_file}:ro")
done

echo "   CUDA mounts : ${#CUDA_MOUNTS[@]} paths mounted from host"

# ── stop any existing instance ────────────────────────────────────────────────
podman rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── run ───────────────────────────────────────────────────────────────────────
echo "► Starting container '${CONTAINER_NAME}' …"
echo "   Input       : ${CAMERA_DEVICE}"
[[ "${CAMERA_DEVICE}" != *://* ]] && echo "   Resolution  : ${CAMERA_WIDTH}×${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps  (local device)"
echo "   Model       : YOLOv8${MODEL_SIZE}  (TensorRT: ${USE_TENSORRT})"
echo "   Confidence  : ${CONFIDENCE}"
echo "   RTSP output : rtsp://${HOST_IP}:${RTSP_OUT_PORT}${RTSP_OUT_PATH}"
echo "   Web preview : http://${HOST_IP}:${WEB_PORT}"

# Build optional group-add flag only if we found the video GID
GROUP_ADD=()
[[ -n "${VIDEO_GID}" ]] && GROUP_ADD+=("--group-add" "${VIDEO_GID}")

# Pass --device only for local device paths, not for URLs
DEVICE_ARG=()
[[ "${CAMERA_DEVICE}" != *://* ]] && DEVICE_ARG+=("--device" "${CAMERA_DEVICE}")

podman run \
  --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  \
  --privileged \
  --user root \
  "${GROUP_ADD[@]}" \
  \
  "${DEVICE_ARG[@]}" \
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
  --env LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib64/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
  --env GST_DEBUG="${GST_DEBUG:-1}" \
  \
  "${IMAGE_NAME}:latest"

echo ""
echo "✓ Container started."
echo ""
echo "  RTSP stream : rtsp://${HOST_IP}:${RTSP_OUT_PORT}${RTSP_OUT_PATH}"
echo "              Play with:  ffplay rtsp://${HOST_IP}:${RTSP_OUT_PORT}${RTSP_OUT_PATH}"
echo "              Or:         vlc    rtsp://${HOST_IP}:${RTSP_OUT_PORT}${RTSP_OUT_PATH}"
echo ""
echo "  Web preview : http://${HOST_IP}:${WEB_PORT}"
echo ""
echo "  Stop:    podman stop ${CONTAINER_NAME}"
echo "  Remove:  podman rm ${CONTAINER_NAME}"
echo ""
echo "► Tailing logs (Ctrl-C to stop following, container keeps running):"
echo "  To increase GStreamer verbosity: GST_DEBUG=3 ./run.sh"
echo ""
podman logs -f "${CONTAINER_NAME}"

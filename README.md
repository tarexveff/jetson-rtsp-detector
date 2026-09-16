# Object Detector — Jetson Orin Nano

A containerised live object-detection application for the **NVIDIA Jetson Orin Nano**.
It ingests video from a **URL** (RTSP, RTMP, HTTP/MJPEG) or a **local V4L2 device** (`/dev/videoN`), runs **YOLOv8** inference on the Jetson GPU, overlays labelled bounding boxes (class name + confidence), and streams the annotated video to a web browser and an RTSP output over your local network.

```
RTSP/HTTP URL  ─┐
                ├──► OpenCV / FFMPEG decode ──► YOLOv8 (CUDA / TensorRT)
/dev/videoN  ──┘    (hw-accelerated where available)      │
                                                 Draw boxes + labels
                                                           │
                                         Flask MJPEG stream ──► Browser
                                         GStreamer RTSP out  ──► VLC/ffplay
```

---

## Requirements

| Component | Version |
|---|---|
| Jetson Orin Nano | JetPack 6.x (CUDA 12.2) or JetPack 5.1.x (CUDA 11.4) |
| Podman | ≥ 4.x |
| Host OS | Ubuntu 22.04 (L4T) **or** RHEL / RHEL-compatible (aarch64) |

> **JetPack 5 users** – change the `FROM` line in [`Dockerfile`](Dockerfile) to  
> `FROM nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3`

---

## Project layout

```
jetson-rtsp-detector/
├── Dockerfile              # NVIDIA L4T-based image, ARM64, CUDA + GStreamer
├── requirements.txt        # Python deps (Flask, ultralytics, OpenCV, …)
├── docker-compose.yml      # podman-compose / docker compose definition
├── run.sh                  # convenience build-and-run script
├── models/                 # weight cache – created at runtime
└── app/
    ├── main.py             # Flask web server + MJPEG/stats routes
    ├── detector.py         # YOLOv8 inference thread + annotation
    └── templates/
        └── index.html      # Dark-theme live viewer UI
```

---

## Quick start

### 1 — Clone / copy the project

```bash
git clone <repo-url> jetson-rtsp-detector
cd jetson-rtsp-detector
```

### 2 — Build and run (single command)

```bash
# Local USB / V4L2 camera (default: /dev/video0)
./run.sh

# Specific local device
./run.sh /dev/video2

# RTSP network camera
CAMERA_DEVICE=rtsp://admin:password@192.168.1.50/stream1 ./run.sh

# HTTP/MJPEG stream
CAMERA_DEVICE=http://192.168.1.60/video ./run.sh
```

Open a browser on any device on the same network:

```
http://<jetson-ip>:8080
```

### 3 — Optional: use podman-compose

```bash
pip install podman-compose          # once

# Local camera
export CAMERA_DEVICE=/dev/video0
podman-compose up -d --build

# RTSP network camera
export CAMERA_DEVICE=rtsp://admin:password@192.168.1.50/stream1
podman-compose up -d --build
```

---

## Configuration

All options are passed as environment variables (or edited in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `CAMERA_DEVICE` | `/dev/video0` | Video source — a local V4L2 device (e.g. `/dev/video0`) **or** a URL (`rtsp://`, `rtsps://`, `rtmp://`, `http://`, `https://`) |
| `CAMERA_WIDTH` | `640` | Capture width in pixels (local devices only; ignored for URLs) |
| `CAMERA_HEIGHT` | `480` | Capture height in pixels (local devices only; ignored for URLs) |
| `CAMERA_FPS` | `30` | Capture frame rate (local devices only; ignored for URLs) |
| `CONFIDENCE` | `0.40` | Minimum detection confidence (0–1) |
| `MODEL_SIZE` | `n` | YOLOv8 variant: `n` nano · `s` small · `m` medium · `l` large · `x` xlarge |
| `WEB_PORT` | `8080` | TCP port the web UI is served on |
| `USE_TENSORRT` | `0` | Set to `1` to export and use a TensorRT engine (faster after first-run export) |
| `MODEL_DIR` | `/models` | In-container path for model weights (map to host with a volume) |

### Examples

```bash
# Local webcam, higher accuracy, TensorRT enabled
export CAMERA_DEVICE=/dev/video0
export MODEL_SIZE=s
export CONFIDENCE=0.50
export USE_TENSORRT=1
./run.sh

# RTSP IP camera
export CAMERA_DEVICE=rtsp://admin:pass@192.168.1.50/ch0
export MODEL_SIZE=s
export CONFIDENCE=0.50
./run.sh
```

---

## Web UI

| URL | Description |
|---|---|
| `http://<jetson-ip>:8080/` | Live annotated viewer (dark-theme) |
| `http://<jetson-ip>:8080/stream` | Raw MJPEG stream (embed in any `<img>` tag) |
| `http://<jetson-ip>:8080/snapshot` | Single JPEG still of the latest frame |
| `http://<jetson-ip>:8080/stats` | JSON: fps, resolution, model, detections |

The viewer page auto-reconnects if the stream is interrupted and includes a one-click **Save Snapshot** button.

---

## GPU passthrough details

The container runs with `privileged: true`, which grants access to all host device nodes. This is the most reliable approach on Jetson because the SoC GPU device tree varies across JetPack versions and across Ubuntu and RHEL host configurations — passing individual `--device` flags is fragile by comparison.

Device nodes exposed to the container include:

- **CUDA compute** – `nvidia0`, `nvidiactl`, `nvidia-uvm`
- **Hardware video decode** – `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`, `nvhost-vic`, `nvhost-nvdla0/1`, `nvmap`

The GStreamer pipeline inside the container uses `nvv4l2decoder` for zero-copy H.264 hardware decoding, keeping CPU usage low and latency minimal.

---

## RHEL support

The project runs on RHEL (and RHEL-compatible) hosts on the Jetson in addition to Ubuntu L4T.

**Key differences on RHEL:**

- **CUDA library path** — On RHEL aarch64 Jetson hosts the NVIDIA CUDA libraries are installed under `/usr/lib64/nvidia` rather than the Ubuntu default of `/usr/lib/aarch64-linux-gnu`.  The [`docker-compose.yml`](docker-compose.yml) bind-mounts `/usr/lib64/nvidia` into the container read-only and sets `LD_LIBRARY_PATH=/usr/lib64/nvidia` so the container finds them automatically.  If your RHEL host installs them to a different path, update the `volumes:` entry and `LD_LIBRARY_PATH` value in [`docker-compose.yml`](docker-compose.yml) to match.

- **SELinux** — The `:z` flag on the `./models` volume mount in [`docker-compose.yml`](docker-compose.yml) relabels the directory for shared container access, which is required when SELinux is enforcing.

- **Podman vs Docker** — RHEL ships Podman by default; `podman-compose` or `docker compose` (via the Docker CE repo) both work.  All commands in this README that use `podman` are interchangeable with `docker`.

- **Container runtime** — No NVIDIA container toolkit is required.  `privileged: true` is sufficient to expose the Jetson's integrated GPU to the container on both Ubuntu and RHEL hosts.

---

## TensorRT acceleration

Setting `USE_TENSORRT=1` causes the detector to:

1. Download the selected YOLOv8 `.pt` weights on first run.
2. Export a FP16 TensorRT `.engine` file (takes ~5–10 minutes).
3. Cache the engine under the `models/` volume.
4. Use the engine for all subsequent inference calls (typically **2–4× faster** than PyTorch).

The engine is only rebuilt if deleted from the volume.

---

## Useful commands

```bash
# Follow logs
podman logs -f rtsp-detector

# Stop
podman stop rtsp-detector

# Remove container (keeps image and model cache)
podman rm rtsp-detector

# Rebuild image after code changes
podman build -t rtsp-detector:latest .

# Check GPU visibility inside the container
podman exec -it rtsp-detector python3 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open network camera: <url>` | Verify the URL is reachable from the Jetson: `ffplay <url>` — check credentials and network connectivity |
| `Cannot open USB camera: /dev/videoN` | Confirm the device node exists (`ls /dev/video*`) and is passed to the container via `--device` |
| `GStreamer pipeline failed; falling back to FFMPEG` | `nvv4l2decoder` not available in container – image may not match your JetPack; check base image tag |
| Browser shows grey "Connecting…" | Container is still loading the model (~30 s on first run); wait and refresh |
| Low FPS with large model | Use a smaller `MODEL_SIZE` (e.g. `n` or `s`), or enable `USE_TENSORRT=1` |
| `no space left on device` during TRT export | Ensure ≥ 4 GB free on the volume mount path |
| Devices not found (`/dev/nvidia0`) | Run `ls /dev/nvidia*` on host; device nodes are created by the Jetson driver stack on first GPU use – try `nvidia-smi` on the host first |

---

## License

MIT — do whatever you like with this.

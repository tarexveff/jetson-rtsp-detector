# RTSP Object Detector — Jetson Orin Nano

A containerised live object-detection application for the **NVIDIA Jetson Orin Nano**.  
It ingests any RTSP video stream, runs **YOLOv8** inference on the Jetson GPU, overlays labelled bounding boxes (class name + confidence), and streams the annotated video to a web browser over your local network.

```
RTSP Camera ──► GStreamer/nvv4l2 HW decode ──► YOLOv8 (CUDA / TensorRT)
                                                       │
                                             Draw boxes + labels
                                                       │
                                         Flask MJPEG stream ──► Browser
```

---

## Requirements

| Component | Version |
|---|---|
| Jetson Orin Nano | JetPack 6.x (CUDA 12.2) or JetPack 5.1.x (CUDA 11.4) |
| Podman | ≥ 4.x |
| Host OS | Ubuntu 22.04 (L4T) |

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
# Replace the URL with your camera's RTSP address
./run.sh rtsp://admin:password@192.168.1.50/stream1
```

Open a browser on any device on the same network:

```
http://<jetson-ip>:8080
```

### 3 — Optional: use podman-compose

```bash
pip install podman-compose          # once
export RTSP_URL=rtsp://admin:password@192.168.1.50/stream1
podman-compose up -d --build
```

---

## Configuration

All options are passed as environment variables (or edited in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | *(required)* | Full RTSP URL including credentials |
| `CONFIDENCE` | `0.40` | Minimum detection confidence (0–1) |
| `MODEL_SIZE` | `n` | YOLOv8 variant: `n` nano · `s` small · `m` medium · `l` large · `x` xlarge |
| `WEB_PORT` | `8080` | TCP port the web UI is served on |
| `USE_TENSORRT` | `0` | Set to `1` to export and use a TensorRT engine (faster after first-run export) |
| `MODEL_DIR` | `/models` | In-container path for model weights (map to host with a volume) |

### Example – higher accuracy, TensorRT enabled

```bash
export RTSP_URL=rtsp://admin:pass@192.168.1.50/ch0
export MODEL_SIZE=s
export CONFIDENCE=0.50
export USE_TENSORRT=1
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

The container uses Podman's `--device` flags to expose every Jetson device node needed for:

- **CUDA compute** – `nvidia0`, `nvidiactl`, `nvidia-uvm`  
- **Hardware video decode** – `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`, `nvhost-vic`, `nvhost-nvdla0/1`, `nvmap`

The GStreamer pipeline inside the container uses `nvv4l2decoder` for zero-copy H.264 hardware decoding, keeping CPU usage low and latency minimal.

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
| `Cannot open RTSP stream` | Verify `RTSP_URL` is reachable from the Jetson: `ffplay <url>` |
| `GStreamer pipeline failed; falling back to FFMPEG` | `nvv4l2decoder` not available in container – image may not match your JetPack; check base image tag |
| Browser shows grey "Connecting…" | Container is still loading the model (~30 s on first run); wait and refresh |
| Low FPS with large model | Use a smaller `MODEL_SIZE` (e.g. `n` or `s`), or enable `USE_TENSORRT=1` |
| `no space left on device` during TRT export | Ensure ≥ 4 GB free on the volume mount path |
| Devices not found (`/dev/nvidia0`) | Run `ls /dev/nvidia*` on host; device nodes are created by the Jetson driver stack on first GPU use – try `nvidia-smi` on the host first |

---

## License

MIT — do whatever you like with this.

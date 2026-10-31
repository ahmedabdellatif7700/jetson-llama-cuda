# 🚀 Jetson Llama CUDA 🦙🔥

**Running local LLMs on a tiny Jetson, because apparently 8 GB of RAM wasn't challenging enough.** 🤖

A CUDA-accelerated [llama.cpp](https://github.com/ggml-org/llama.cpp) Docker environment for the NVIDIA Jetson Orin Nano.

## 🧪 Tested Environment

- 🖥️ **Hardware:** NVIDIA Jetson Orin Nano 8GB
- 🐧 **OS:** Ubuntu 24.04 (AArch64)
- ⚡ **CUDA:** 13.2
- 🧠 **GPU architecture:** SM 8.7
- 🦙 **llama.cpp:** Commit `5047748f1`
- 🤖 **Models:** Llama 3.2 1B Instruct (default), Qwen2.5-Coder 7B Instruct (both Q4_K_M)

## 🛠️ Features

- 🚀 CUDA-accelerated llama.cpp inference
- 🌐 OpenAI-compatible REST API
- 📌 Pinned llama.cpp revision for reproducibility
- 🔗 CUDA linking stub provided by the NVIDIA base image
- 🩹 Workaround for the NVIDIA Container Toolkit 1.19.1 CDI compatibility-hook crash
- 📦 Dockerized setup without bundling model weights

## 🏗️ Build — Time for Coffee ☕

```bash
docker build -t jetson-llama-cuda:local .
```

The initial CUDA compilation can take a while. Seriously, go make some coffee. ☕🐢

## 📁 Models — Feed the Llama 🦙

The default model is **Llama 3.2 1B Instruct (Q4_K_M, ~0.8 GB)**. Download it into `models/`:

```bash
mkdir -p models
curl -L -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

To use a different GGUF model, put it in `models/` and select it in `.env` (see `.env.example`):

```bash
MODEL_FILE=qwen2.5-coder-7b-instruct-q4_k_m.gguf
MODEL_NAME=Qwen2.5-Coder 7B Instruct
CONTEXT_SIZE=2048
```

Then run `make restart`. `make models` lists what's downloaded and marks the active model.

Model files are excluded from Git because GitHub probably doesn't want your multi-gigabyte llama lunch. 🍽️

## 🚀 Run — Wake Up the Llama!

```bash
docker run --rm -it \
  --name llama-server \
  --runtime=nvidia --gpus all \
  --network host \
  -v "$PWD/models:/models:ro" \
  jetson-llama-cuda:local \
  -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -ngl 99 -c 4096 \
  --host 127.0.0.1 --port 8080
```

The server runs in the foreground. Press `Ctrl+C` when the llama has worked hard enough.

## 🐳 Docker Compose — API + Model in One Command

`docker-compose.yml` runs two services on a private network:

- **`llama`** — the CUDA image above, `llama-server` bound to `0.0.0.0:8080` *inside* the compose network only (no host port published)
- **`api`** — a small FastAPI front-end (`api/`) published on `127.0.0.1:8000`, the only exposed port

```bash
docker compose up --build
```

Both services have healthchecks. `api` waits until `llama` reports healthy (model fully loaded, up to a 3-minute start period), so the first requests don't hit a half-started server. Check progress with `docker compose ps`.

Optionally set an API key (the `api` service then requires an `X-API-Key` header on `/chat` and `/v1/*`):

```bash
cp .env.example .env   # then edit .env and set API_KEY=...
```

### API service endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Public readiness check: `200` once the model is loaded, `503` otherwise (no API key needed) |
| `POST /chat` | Simple prompt in, text out: `{"prompt": "...", "system": "...", "max_tokens": 256, "temperature": 0.7}` |
| `/v1/*` | Transparent proxy to llama-server's OpenAI-compatible API, including SSE streaming |

```bash
# Simple chat
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt": "Explain CUDA warps briefly.", "max_tokens": 100}'

# Full OpenAI-compatible endpoint (proxied, streaming supported)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain CUDA warps briefly."}
    ],
    "max_tokens": 100
  }'
```

No cloud required. Just a Jetson, some CUDA cores, and a surprisingly talkative model.

## 💻 From Your Laptop — SSH Tunnel

The API only listens on the Jetson's localhost, so reach it through an SSH tunnel. Clone this repo on the laptop too (only `python3`, `ssh` and `make` are needed there), then:

```bash
make jetson-up       # start the stack on the Jetson over SSH (no-op if already running)
make tunnel-start    # background tunnel: laptop localhost:8000 -> Jetson API
make chat            # interactive chat (or: make ask PROMPT="...", make smoke)
make tunnel-stop     # close the tunnel
```

The SSH host defaults to `orinnano8s` (from your `~/.ssh/config`); override with `make tunnel-start JETSON=user@host`. If port 8000 is taken on the laptop, add `LOCAL_PORT=8001` to every command. Run `make` for the full list.

## 🌐 REST API — Talk to the Machine Directly 🤖

Without compose, `llama-server` itself speaks the OpenAI-compatible REST API on port 8080 (see the `docker run` example above):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain CUDA warps briefly."}
    ],
    "max_tokens": 100
  }'
```

## 📊 Performance — Tiny Board, Big Ambitions 💪

Measured on a Jetson Orin Nano 8GB, all layers on the GPU, Q4_K_M quantization:

| Metric | Llama 3.2 1B Instruct | Qwen2.5-Coder 7B Instruct |
|---|---|---|
| 📦 Model file | ~0.8 GB | ~4.7 GB |
| ⚡ Prompt processing | ~565 tokens/s | ~173 tokens/s |
| 🦙 Token generation | ~49 tokens/s | ~11.7 tokens/s |
| 🧠 Context | 4096 tokens | 2048 tokens |
| ⏱️ Model load | ~5 s | ~10 s |

These are illustrative measurements, not standardized benchmarks. Actual performance depends on model, workload, memory, and device configuration.

## 🩹 CUDA Compatibility — The Bug That Wanted Attention 🐛

NVIDIA Container Toolkit 1.19.1 encountered a CDI compatibility-hook crash when processing CUDA forward-compatibility libraries in this environment.

The Dockerfile works around this by disabling the affected compatibility directories **inside the image**, allowing the NVIDIA container runtime to supply the Jetson host driver.

No host driver modifications are required.

## 🔐 Security — Don't Let Strangers Feed Your Llama

Both the compose API (published on `127.0.0.1:8000`) and the direct server example (on `127.0.0.1:8080`) bind to localhost only.

With compose, set `API_KEY` in `.env` to require the `X-API-Key` header; the raw `llama-server` port is not published to the host at all.

Before exposing either to other machines or networks, configure appropriate authentication and network access controls.

## 🎯 Project Status

- ✅ CUDA Docker image builds
- ✅ NVIDIA container runtime starts successfully
- ✅ CUDA linking stub resolved from base image
- ✅ Llama 3.2 1B and Qwen2.5-Coder 7B inference tested
- ✅ Dockerized REST API front-end (`docker compose up`) with health check and optional API key
- 🚧 Further GPU profiling and performance optimization

---

**Built for curiosity, CUDA experiments, and the occasional debugging marathon.** 🦙⚡🐧

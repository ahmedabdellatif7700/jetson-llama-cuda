# 🚀 Jetson Llama CUDA 🦙🔥

**Running local LLMs on a tiny Jetson, because apparently 8 GB of RAM wasn't challenging enough.** 🤖

A CUDA-accelerated [llama.cpp](https://github.com/ggml-org/llama.cpp) Docker environment for the NVIDIA Jetson Orin Nano.

## 🧪 Tested Environment

- 🖥️ **Hardware:** NVIDIA Jetson Orin Nano 8GB
- 🐧 **OS:** Ubuntu 24.04 (AArch64)
- ⚡ **CUDA:** 13.2
- 🧠 **GPU architecture:** SM 8.7
- 🦙 **llama.cpp:** Commit `5047748f1`
- 🤖 **Models:** SmolLM2 135M Instruct (default, FP16); Llama 3.2 1B and Qwen2.5-Coder 7B (Q4_K_M)

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

The default model is **SmolLM2 135M Instruct (FP16, ~271 MB)**. Download its
GGUF conversion into `models/` (the original trainable Safetensors weights are
kept separately in `models/SmolLM2-135M-Instruct/`):

```bash
mkdir -p models
curl -fL -o models/SmolLM2-135M-Instruct-f16.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-f16.gguf
echo "f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57  models/SmolLM2-135M-Instruct-f16.gguf" | sha256sum -c -
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
  -m /models/SmolLM2-135M-Instruct-f16.gguf \
  -ngl 99 -c 4096 \
  --host 127.0.0.1 --port 8080
```

The server runs in the foreground. Press `Ctrl+C` when the llama has worked hard enough.

## 🐳 Docker Compose — Open WebUI + llama.cpp

The CUDA `llama` service is unchanged. `open-webui` connects directly to
`http://llama:8080/v1` on the private Compose network. Only WebUI is published,
on port **3000**, accessible at `http://<jetson-private-or-vpn-ip>:3000`.
Create your administrator account on the first visit; authentication stays enabled.

The official ARM64-compatible `main-slim` image is pinned by digest. It contains
no Ollama or local ML stack. Offline mode blocks model downloads. One WebUI worker
and a 1 GB memory cap leave room for CUDA inference in the Jetson's shared RAM.
Built-in agent tools are disabled to keep their schemas out of the 4096-token
chat context. Token usage reporting is enabled; use the information icon under a
new reply to inspect input/output counts. A short default system instruction
identifies the local Jetson deployment. The existing model/context selection is unchanged. Local RAG and voice models
are not installed; this configuration is intended for text chat with llama.cpp.

Conversations, accounts, settings, and the generated WebUI secret persist in the
`open-webui` named volume mounted at `/app/backend/data`. Preserve that volume.
Admin-panel connection changes persist and take precedence over initial environment
values. The initial OpenAI connection URL is `http://llama:8080/v1`, key `unused`.

```bash
# Existing installation: pull/start only WebUI without restarting llama.cpp.
docker compose pull open-webui
docker compose up -d --no-deps --no-build open-webui
# Stop/remove the retired API container only (preserves images and volumes):
docker stop jetson-llama-cuda-api-1
docker rm jetson-llama-cuda-api-1

# On the Jetson: start both services (existing llama image required):
make
# Stop both services, preserving all data:
make stop
# Restart when needed:
make up
make smoke
```

`make down` and `make clean` preserve images, volumes, and model files.
Do not use `down -v`, image/volume pruning, or image-removal options if you want
to preserve them. Legacy `api/` and `scripts/llm.py` remain on disk but are no longer
part of the running stack; their custom `/chat` endpoint and API key do not apply.

Port 3000 binds all interfaces by default for private LAN/VPN access. Optionally
set `WEBUI_BIND_ADDRESS` in `.env` to a specific private or VPN interface address.
To access through SSH instead:

```bash
make tunnel-start JETSON=user@jetson
# Open http://127.0.0.1:3000
make tunnel-stop
```

`make smoke` checks model discovery, completions and SSE streaming from within
WebUI's container, plus the published health and HTML endpoints. It creates no
accounts. After signup, select the discovered model and send a chat in the browser
to confirm the authenticated UI flow from your client network.

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

## Fine-tune SmolLM2 locally

A separate [LoRA training workflow](training/README.md) uses the original
SmolLM2 Safetensors weights on the Jetson GPU. Run `make train-setup` once,
then `make train` for a small demonstration. Alternatively, run `cd training`
and `make` (or `make train`) using that directory's own Makefile. Training uses
explicit JSONL datasets, not WebUI conversations. Training stops inference to free
memory; run `make` afterwards to return to WebUI. Saved adapters never overwrite
the deployed GGUF or original weights.

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

Open WebUI requires a login. Its port 3000 is published for private LAN/VPN
access; llama.cpp has no host port. Keep access within your intended private
network. The old custom API's `X-API-Key` is no longer used.

## 🎯 Project Status

- ✅ CUDA Docker image builds
- ✅ NVIDIA container runtime starts successfully
- ✅ CUDA linking stub resolved from base image
- ✅ Llama 3.2 1B and Qwen2.5-Coder 7B inference tested
- ✅ Open WebUI with persistent chats and direct llama.cpp connectivity
- 🚧 Further GPU profiling and performance optimization

---

**Built for curiosity, CUDA experiments, and the occasional debugging marathon.** 🦙⚡🐧

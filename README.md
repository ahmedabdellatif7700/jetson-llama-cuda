# 🚀 Jetson Llama CUDA 🦙🔥

**Running local LLMs on a tiny Jetson, because apparently 8 GB of RAM wasn't challenging enough.** 🤖

A CUDA-accelerated [llama.cpp](https://github.com/ggml-org/llama.cpp) Docker environment for the NVIDIA Jetson Orin Nano.

## 🧪 Tested Environment

- 🖥️ **Hardware:** NVIDIA Jetson Orin Nano 8GB
- 🐧 **OS:** Ubuntu 24.04 (AArch64)
- ⚡ **CUDA:** 13.2
- 🧠 **GPU architecture:** SM 8.7
- 🦙 **llama.cpp:** Commit `5047748f1`
- 🤖 **Model:** Qwen2.5-Coder 7B Instruct (Q4_K_M)

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

Create a model directory:

```bash
mkdir -p models
```

Place your GGUF model inside `models/`.

Model files are excluded from Git because GitHub probably doesn't want your multi-gigabyte llama lunch. 🍽️

## 🚀 Run — Wake Up the Llama!

```bash
docker run --rm -it \
  --name qwen-server \
  --runtime=nvidia --gpus all \
  --network host \
  -v "$PWD/models:/models:ro" \
  jetson-llama-cuda:local \
  -m /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -ngl 99 -c 2048 \
  --host 127.0.0.1 --port 8080
```

The server runs in the foreground. Press `Ctrl+C` when the llama has worked hard enough.

## 🌐 REST API — Talk to the Machine 🤖

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

No cloud required. Just a Jetson, some CUDA cores, and a surprisingly talkative model.

## 📊 Performance — Tiny Board, Big Ambitions 💪

Measured using Qwen2.5-Coder 7B Instruct Q4_K_M:

| Metric | Observed |
|---|---|
| ⚡ Prompt processing | ~173 tokens/s |
| 🦙 Token generation | ~11.7 tokens/s |
| 🧠 Context per slot | 2048 tokens |
| 🖥️ Platform | Jetson Orin Nano 8GB |

These are illustrative measurements, not standardized benchmarks. Actual performance depends on model, workload, memory, and device configuration.

## 🩹 CUDA Compatibility — The Bug That Wanted Attention 🐛

NVIDIA Container Toolkit 1.19.1 encountered a CDI compatibility-hook crash when processing CUDA forward-compatibility libraries in this environment.

The Dockerfile works around this by disabling the affected compatibility directories **inside the image**, allowing the NVIDIA container runtime to supply the Jetson host driver.

No host driver modifications are required.

## 🔐 Security — Don't Let Strangers Feed Your Llama

The example server listens on `127.0.0.1` without an API key.

Before exposing it to other machines or networks, configure appropriate authentication and network access controls.

## 🎯 Project Status

- ✅ CUDA Docker image builds
- ✅ NVIDIA container runtime starts successfully
- ✅ CUDA linking stub resolved from base image
- ✅ Qwen2.5-Coder 7B inference tested
- 🚧 Further GPU profiling and performance optimization

---

**Built for curiosity, CUDA experiments, and the occasional debugging marathon.** 🦙⚡🐧

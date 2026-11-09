# SmolLM2 LoRA training on the Jetson

This trains a small LoRA adapter using the original local Safetensors weights.
It does not train the GGUF file, learn automatically from WebUI chats, or replace
what WebUI serves. Original model files and existing adapters are preserved.

## Start here: the Python training directory

`training/` contains the Python fine-tuning workflow and its own Makefile:

| File | Purpose |
|---|---|
| `Makefile` | Runs setup, dataset training, and adapter preview |
| `train_lora.py` | Trains and saves a LoRA adapter on CUDA |
| `check_cuda.py` | Verifies GPU forward/backward calculations |
| `preview.py` | Loads a saved adapter and generates a test answer |
| `demo.jsonl` / `demo-validation.jsonl` | Explicit demonstration datasets |
| `requirements.txt` | Pinned training libraries |
| `data/` / `runs/` | Private datasets / generated adapters (ignored by Git) |

On the Jetson:

```bash
cd ~/jetson-llama-cuda/training
make setup             # Once; already installed on this Jetson
make                   # Train from the demo dataset
# Equivalent: make train

# Train from your own datasets; paths are relative to this directory:
make train DATA=data/train.jsonl VALIDATION=data/validation.jsonl STEPS=100

# Test a saved adapter:
make preview ADAPTER=runs/cedar-demo-001

# Resume the existing WebUI inference services:
make -C ..
```

Training reads only the explicitly selected JSONL datasets. It does not read your
WebUI conversations, replies, or feedback. Each run saves a new adapter separately
from the original weights and deployed GGUF.

**The directory matters:** plain `make` in the repository root starts inference;
plain `make` in `training/` trains an adapter. Both `make train` in the root and
`make train` in this directory use the same training implementation. Training and
preview stop inference containers to free shared memory; they do not restart them
automatically. The root-directory commands below remain available as shortcuts.

The setup uses Python 3.12, the official ARM64 PyTorch CUDA 13.2 wheel, Transformers,
and PEFT in `.venv-training`. It does not change system Python or Docker images.
The initial dependency download is several GB; model training is offline.

The tested PyTorch wheel emits a warning that SM 8.7 is not an explicit build
target. The CUDA matrix backward check and the actual eager SmolLM2 LoRA run
are tested on this board; that does not guarantee support for other PyTorch
operations or models. Keep the CUDA preflight check enabled.

```bash
make train-setup       # Once: install isolated dependencies; verify CUDA backward pass
make train            # Stop inference services, then train the demonstration adapter
make                  # Start the existing inference model again
```

`make train` stops both containers to free unified memory. It leaves them stopped
even if training fails; run `make` when finished. Defaults: 40 optimizer steps,
batch size 1 with 4 gradient accumulation steps, 256-token maximum example length,
FP32 weights, rank-8 LoRA on attention q/v projections, and AdamW. No bitsandbytes,
FlashAttention, or external training service is required. Loss is computed only
on the final assistant answer, including its end-of-turn marker.

The included Cedar robot dataset is fictional and is only a plumbing demonstration.
Its held-out prompts are simple paraphrases, not evidence of general model quality.
A lower validation loss is useful but does not establish reliable factual recall.
Choose examples that match the actual behavior you want to teach.

## How Python training connects to inference

Training is written in Python in [train_lora.py](train_lora.py). Python controls
loading examples, calculating loss, updating the adapter, and saving results.
PyTorch performs the tensor calculations on the Jetson GPU through CUDA.
Transformers loads the model and tokenizer; PEFT adds the trainable LoRA adapter.

```text
Original SmolLM2 Safetensors weights + training examples (JSONL)
                            |
                            v
              Python -> PyTorch -> CUDA GPU
                            |
                            v
              Saved LoRA adapter (learned changes)
```

The base model weights stay frozen during LoRA training. The adapter contains
small learned changes and needs the original model to generate answers.
[preview.py](preview.py) loads both together to test the result on the GPU.
The Makefile connects the commands: `make train` runs the training script, and
`make train-preview ADAPTER=...` runs the preview script in `.venv-training`.

Browser chat uses a separate inference path:

```text
Browser -> Open WebUI -> llama.cpp OpenAI-compatible API -> GGUF model
                                    (port 8080)           on Jetson GPU
```

The trained adapter is not connected to WebUI automatically. WebUI continues
using the original GGUF until an explicit deployment step. To serve the trained
result, export the adapter to a llama.cpp-compatible GGUF adapter and configure
llama.cpp to load it with the matching base model, or merge the adapter into the
original Safetensors weights and convert the merged model to GGUF. Architecture
and conversion-tool support must be verified for the chosen export path.

This repository currently provides training and Python preview; adapter export
and deployment into WebUI are not implemented yet. Saving WebUI conversations
alone does not update either the base model or an adapter.

## Your own data

Put private datasets under `training/data/` (ignored by Git). Use one JSON object
per line, with a `messages` array ending in a nonempty assistant answer:

```json
{"messages":[{"role":"user","content":"What is the robot called?"},{"role":"assistant","content":"The robot is named Cedar."}]}
```

Use separate training and held-out validation examples. Long examples fail with
an actionable error rather than silently dropping the assistant answer.

```bash
make train DATA=training/data/train.jsonl VALIDATION=training/data/validation.jsonl STEPS=100
```

Each run creates a new timestamped directory under `training/runs/`. Override with
`OUTPUT=training/runs/my-experiment`; existing directories are rejected. Each output
contains `adapter_model.safetensors`, adapter/tokenizer configuration, and a report
with dataset hashes, losses before/after, GPU details and peak allocated GPU memory.
An adapter contains only learned changes and requires the original base weights.

## Inspect the trained adapter

```bash
make train-preview ADAPTER=training/runs/<run-directory> PROMPT="What is the robot called?"
```

Preview also stops inference services to leave memory for PyTorch. It loads the
adapter with the original base model and generates locally on CUDA. Run `make`
to return to WebUI. Previewing does not deploy or alter the inference GGUF.

Deploying an adapter in llama.cpp needs a separate conversion/export step. Keep the
base Safetensors weights, the adapter, and `training-report.json` together. Do not
point `MODEL_FILE` at an adapter directory; llama.cpp expects a GGUF model.

## Verified demonstration run

On the Orin Nano 8GB, the 40-step demo updated 460,800 adapter parameters.
Held-out token loss decreased from 2.9634 to 0.5655. Peak PyTorch allocated CUDA
memory was about 637 MiB (not total system memory or all CUDA allocations).
The adapter was saved under `training/runs/cedar-demo-001/` on the setup machine.
These generated files are deliberately ignored by Git.

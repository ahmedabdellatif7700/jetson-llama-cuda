# SmolLM2 LoRA training on the Jetson

This trains a small LoRA adapter using the original local Safetensors weights.
It does not train the GGUF file, learn automatically from WebUI chats, or replace
what WebUI serves. Original model files and existing adapters are preserved.

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

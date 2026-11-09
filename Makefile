# Jetson Llama CUDA — common tasks. Run `make` to start; `make stop` to stop; `make help` for the list.
#
# The same Makefile works in both places:
#   on the Jetson  -> manage the docker stack (make up, make down, ...)
#   on the laptop  -> open an SSH tunnel to the Jetson, then open WebUI
# Open the WebUI in your browser after starting the stack or tunnel.

# Model and optional WebUI bind settings come from .env.
-include .env

JETSON     ?= orinnano8s
JETSON_DIR ?= jetson-llama-cuda
LOCAL_PORT ?= 3000
export API_URL ?= http://127.0.0.1:$(LOCAL_PORT)

COMPOSE := docker compose
# SSH control socket so the background tunnel can be stopped cleanly.
TUNNEL_SOCKET := /tmp/jetson-llama-tunnel-$(JETSON).sock

.DEFAULT_GOAL := up
.PHONY: help build up stop down restart logs ps wait models \
        tunnel tunnel-start tunnel-stop tunnel-status jetson-up jetson-down jetson-ps jetson-logs \
        health chat smoke test clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-z-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ On the Jetson: manage the stack

build: ## Build the llama.cpp image (slow; unnecessary for WebUI migration)
	$(COMPOSE) build

up: ## Start the stack in the background and wait until the model is loaded
	$(COMPOSE) up -d --no-build --wait

stop: ## Stop both services; preserve containers, images, models, and conversations
	$(COMPOSE) stop

down: ## Stop and remove the containers
	$(COMPOSE) down

restart: down up ## Restart the stack

logs: ## Follow logs from both services
	$(COMPOSE) logs -f

ps: ## Show container status and health
	$(COMPOSE) ps

wait: ## Wait for llama.cpp and Open WebUI healthchecks
	$(COMPOSE) up -d --no-build --wait

models: ## List downloaded models (* = selected by MODEL_FILE in .env)
	@active=$$($(COMPOSE) config | awk '/- \/models\// {sub(".*/models/", ""); print; exit}'); \
	for f in models/*.gguf; do \
		name=$$(basename "$$f"); \
		[ "$$name" = "$$active" ] && mark='*' || mark=' '; \
		printf ' %s %-45s %s\n' "$$mark" "$$name" "$$(du -h "$$f" | cut -f1)"; \
	done

test: up smoke ## Bring the stack up, then run the smoke test

clean: ## Stop containers while preserving images, models, and volumes
	$(COMPOSE) down

##@ On the laptop: reach the Jetson (JETSON=orinnano8s, LOCAL_PORT=3000)

tunnel: ## Open the SSH tunnel in the foreground (Ctrl+C to close)
	ssh -N -L $(LOCAL_PORT):127.0.0.1:3000 $(JETSON)

tunnel-start: ## Open the SSH tunnel in the background
	ssh -f -N -M -S $(TUNNEL_SOCKET) -o ExitOnForwardFailure=yes \
		-L $(LOCAL_PORT):127.0.0.1:3000 $(JETSON)
	@echo "tunnel open: $(API_URL) -> $(JETSON):3000"

tunnel-stop: ## Close the background SSH tunnel
	ssh -S $(TUNNEL_SOCKET) -O exit $(JETSON)

tunnel-status: ## Check whether the background tunnel is open
	ssh -S $(TUNNEL_SOCKET) -O check $(JETSON)

jetson-up: ## Start the stack on the Jetson remotely and wait for the model
	ssh $(JETSON) 'cd $(JETSON_DIR) && make up'

jetson-down: ## Stop the stack on the Jetson remotely (frees GPU memory)
	ssh $(JETSON) 'cd $(JETSON_DIR) && make down'

jetson-ps: ## Show container status on the Jetson
	ssh $(JETSON) 'cd $(JETSON_DIR) && make -s ps'

jetson-logs: ## Follow the Jetson's container logs
	ssh -t $(JETSON) 'cd $(JETSON_DIR) && make logs'

##@ Browser access

health: ## Check Open WebUI readiness
	@curl -fsS $(API_URL)/health

chat: ## Show the browser URL
	@echo "Open $(API_URL) in your browser"

smoke: ## Verify discovery, completions, streaming, and WebUI HTTP access
	@python3 scripts/verify_webui.py

##@ On the Jetson: offline LoRA training (separate from inference)

TRAIN_PY := .venv-training/bin/python
DATA ?= training/demo.jsonl
VALIDATION ?= training/demo-validation.jsonl
STEPS ?= 40
OUTPUT ?= training/runs/$(shell date -u +%Y%m%dT%H%M%SZ)
PROMPT ?= What is the demonstration robot called?
.PHONY: train-setup train train-preview

train-setup: ## Install isolated CUDA training dependencies and verify backward pass
	python3 -m venv .venv-training
	$(TRAIN_PY) -m pip install 'torch==2.12.1+cu132' --index-url https://download.pytorch.org/whl/cu132
	$(TRAIN_PY) -m pip install -r training/requirements.txt
	$(TRAIN_PY) training/check_cuda.py

train: ## Stop inference and train a new LoRA adapter; run make afterwards to resume WebUI
	@test -x $(TRAIN_PY) || { echo "Run make train-setup first"; exit 1; }
	@test -f "$(DATA)" -a -f "$(VALIDATION)" || { echo "Training/validation data missing"; exit 1; }
	@test ! -e "$(OUTPUT)" || { echo "Output exists; choose a new OUTPUT"; exit 1; }
	$(TRAIN_PY) training/check_cuda.py
	$(COMPOSE) stop
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(TRAIN_PY) training/train_lora.py --data "$(DATA)" --validation "$(VALIDATION)" --steps "$(STEPS)" --output "$(OUTPUT)"

train-preview: ## Preview ADAPTER=training/runs/<run>; stops inference until make
	@test -n "$(ADAPTER)" -a -f "$(ADAPTER)/adapter_config.json" || { echo "Set ADAPTER to a saved adapter directory"; exit 1; }
	$(COMPOSE) stop
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(TRAIN_PY) training/preview.py --adapter "$(ADAPTER)" --prompt "$(PROMPT)"

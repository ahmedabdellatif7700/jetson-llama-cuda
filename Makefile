# Jetson Llama CUDA — common tasks. Run `make` or `make help` for the list.
#
# The same Makefile works in both places:
#   on the Jetson  -> manage the docker stack (make up, make down, ...)
#   on the laptop  -> open an SSH tunnel to the Jetson, then talk to the API
# The client targets (chat, ask, health, smoke) work on either side.

# API_KEY (if any) comes from .env so the client matches the api service.
-include .env
export API_KEY

JETSON     ?= orinnano8s
JETSON_DIR ?= jetson-llama-cuda
LOCAL_PORT ?= 8000
export API_URL ?= http://127.0.0.1:$(LOCAL_PORT)

COMPOSE := docker compose
LLM     := python3 scripts/llm.py
PROMPT  ?= Explain CUDA warps briefly.
# SSH control socket so the background tunnel can be stopped cleanly.
TUNNEL_SOCKET := /tmp/jetson-llama-tunnel-$(JETSON).sock

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs ps wait models \
        tunnel tunnel-start tunnel-stop tunnel-status jetson-up jetson-down jetson-ps jetson-logs \
        health ask chat smoke test clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-z-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ On the Jetson: manage the stack

build: ## Build both images (first build compiles llama.cpp; slow)
	$(COMPOSE) build

up: ## Start the stack in the background and wait until the model is loaded
	$(COMPOSE) up -d --build
	@$(LLM) wait

down: ## Stop and remove the containers
	$(COMPOSE) down

restart: down up ## Restart the stack

logs: ## Follow logs from both services
	$(COMPOSE) logs -f

ps: ## Show container status and health
	$(COMPOSE) ps

wait: ## Block until the API reports the model ready
	@$(LLM) wait

models: ## List downloaded models (* = selected by MODEL_FILE in .env)
	@active=$$($(COMPOSE) config | awk '/- \/models\// {sub(".*/models/", ""); print; exit}'); \
	for f in models/*.gguf; do \
		name=$$(basename "$$f"); \
		[ "$$name" = "$$active" ] && mark='*' || mark=' '; \
		printf ' %s %-45s %s\n' "$$mark" "$$name" "$$(du -h "$$f" | cut -f1)"; \
	done

test: up smoke ## Bring the stack up, then run the smoke test

clean: ## Stop the stack and remove the built images (keeps models/)
	$(COMPOSE) down --rmi local

##@ On the laptop: reach the Jetson (JETSON=orinnano8s, LOCAL_PORT=8000)

tunnel: ## Open the SSH tunnel in the foreground (Ctrl+C to close)
	ssh -N -L $(LOCAL_PORT):127.0.0.1:8000 $(JETSON)

tunnel-start: ## Open the SSH tunnel in the background
	ssh -f -N -M -S $(TUNNEL_SOCKET) -o ExitOnForwardFailure=yes \
		-L $(LOCAL_PORT):127.0.0.1:8000 $(JETSON)
	@echo "tunnel open: $(API_URL) -> $(JETSON):8000"

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

##@ Either side: talk to the API at $API_URL

health: ## Query GET /health
	@curl -s -w '\nHTTP %{http_code}\n' $(API_URL)/health

ask: ## One-shot question: make ask PROMPT="..."
	@$(LLM) ask "$(PROMPT)"

chat: ## Interactive streaming chat with the model
	@$(LLM) chat

smoke: ## End-to-end check: health, auth, correctness, streaming, speed
	@$(LLM) smoke

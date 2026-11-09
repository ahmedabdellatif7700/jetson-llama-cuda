# Run make or make train here to fine-tune from a dataset.
# Delegate to the repository targets so CUDA checks and memory handling stay shared.
TRAINING_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(TRAINING_DIR)/..)

.DEFAULT_GOAL := train
.PHONY: train setup train-setup preview train-preview help

DATA ?= $(TRAINING_DIR)/demo.jsonl
VALIDATION ?= $(TRAINING_DIR)/demo-validation.jsonl
STEPS ?= 40
OUTPUT ?= $(TRAINING_DIR)/runs/$(shell date -u +%Y%m%dT%H%M%SZ)
PROMPT ?= What is the demonstration robot called?

train: ## Train from DATA and VALIDATION; stop inference to free GPU memory
	$(MAKE) -C "$(REPO_ROOT)" train DATA="$(abspath $(DATA))" VALIDATION="$(abspath $(VALIDATION))" STEPS="$(STEPS)" OUTPUT="$(abspath $(OUTPUT))"

setup train-setup: ## Install the isolated Python/CUDA dependencies (once)
	$(MAKE) -C "$(REPO_ROOT)" train-setup

preview train-preview: ## Preview a saved adapter: make preview ADAPTER=runs/<run>
	@test -n "$(ADAPTER)" || { echo "Set ADAPTER=runs/<run-directory>"; exit 1; }
	$(MAKE) -C "$(REPO_ROOT)" train-preview ADAPTER="$(abspath $(ADAPTER))" PROMPT="$(PROMPT)"

help: ## Show commands; training consumes JSONL datasets, not WebUI chats
	@echo "make / make train       Train a new adapter from the demo dataset"
	@echo "make setup              Install dependencies and verify CUDA"
	@echo "make preview ADAPTER=runs/<run>  Test a saved adapter"
	@echo "make train DATA=data/train.jsonl VALIDATION=data/validation.jsonl STEPS=100"
	@echo "Training/preview stop inference. Resume WebUI with: make -C .."

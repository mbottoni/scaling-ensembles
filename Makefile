SHELL := /bin/bash

PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 5000
HOST ?= 127.0.0.1
MLFLOW_STORE ?= outputs/mlruns
OUTPUT_ROOT ?= outputs
CONFIG ?= experiments/mnist_width_sweep.yaml

SERIES_CONFIGS := \
	experiments/series/mnist_mlp_extended_width_sweep.yaml \
	experiments/series/cifar10_cnn_extended_width_sweep.yaml \
	experiments/series/cifar10_cnn_matched_train_loss.yaml \
	experiments/series/dataset_difficulty_mnist.yaml \
	experiments/series/dataset_difficulty_fashionmnist.yaml \
	experiments/series/dataset_difficulty_cifar10.yaml \
	experiments/series/cifar10_patch_transformer_long_schedule.yaml \
	experiments/series/cifar10_cnn_gaussian_noise_eval.yaml \
	experiments/series/cifar10_cnn_blur_eval.yaml \
	experiments/series/cifar100_cnn_width_sweep.yaml \
	experiments/series/svhn_cnn_width_sweep.yaml \
	experiments/series/stl10_cnn_width_sweep.yaml \
	experiments/series/cifar10_resnet_width_sweep.yaml

.DEFAULT_GOAL := help

.PHONY: help venv install install-tracking mlflow-ui mlflow-ui-public \
	sweep sweep-mnist sweep-cifar-cnn sweep-cifar-transformer sweep-resnet series \
	analysis analysis-plots plots plots-series marimo report compile clean-cache

help:
	@echo "Useful commands:"
	@echo "  make venv                 Create $(VENV)"
	@echo "  make install              Install package in editable mode"
	@echo "  make install-tracking     Install package with MLflow support"
	@echo "  make mlflow-ui            Start MLflow UI at http://$(HOST):$(PORT)"
	@echo "  make sweep CONFIG=...     Run one experiment config"
	@echo "  make sweep-resnet         Run CIFAR-10 ResNet width sweep"
	@echo "  make series               Run all series experiments with MPS fallback"
	@echo "  make analysis             Run post-hoc analysis on series outputs"
	@echo "  make analysis-plots       Generate analysis plots (ECE, scaling, barriers)"
	@echo "  make plots                Generate paper-style plots from $(OUTPUT_ROOT)"
	@echo "  make plots-series         Generate paper-style plots from outputs/series"
	@echo "  make marimo               Open the Marimo visualization app"
	@echo "  make report               Compile the LaTeX report when latexmk is available"
	@echo "  make compile              Compile Python sources"
	@echo "  make clean-cache          Remove local cached logits"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e .

install-tracking: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[tracking]"

mlflow-ui:
	$(BIN)/mlflow ui --backend-store-uri $(MLFLOW_STORE) --host $(HOST) --port $(PORT)

mlflow-ui-public:
	$(MAKE) mlflow-ui HOST=0.0.0.0

sweep:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(BIN)/scaling-ensembles-sweep --config $(CONFIG)

sweep-mnist:
	$(MAKE) sweep CONFIG=experiments/mnist_width_sweep.yaml

sweep-cifar-cnn:
	$(MAKE) sweep CONFIG=experiments/cifar10_cnn_width_sweep.yaml

sweep-cifar-transformer:
	$(MAKE) sweep CONFIG=experiments/cifar10_patch_transformer_width_sweep.yaml

sweep-resnet:
	$(MAKE) sweep CONFIG=experiments/series/cifar10_resnet_width_sweep.yaml

analysis:
	$(BIN)/python scripts/run_analysis.py

analysis-plots:
	MPLCONFIGDIR=outputs/.matplotlib $(BIN)/python scripts/make_analysis_plots.py
	MPLCONFIGDIR=outputs/.matplotlib $(BIN)/python scripts/make_reliability_diagrams.py

series:
	@set -euo pipefail; \
	for config in $(SERIES_CONFIGS); do \
		echo "===== RUNNING $$config ====="; \
		PYTORCH_ENABLE_MPS_FALLBACK=1 $(BIN)/scaling-ensembles-sweep --config "$$config"; \
	done

plots:
	MPLCONFIGDIR=$(OUTPUT_ROOT)/.matplotlib $(BIN)/scaling-ensembles-paper-plots --output-root $(OUTPUT_ROOT)

plots-series:
	$(MAKE) plots OUTPUT_ROOT=outputs/series

marimo:
	$(BIN)/marimo edit notebooks/width_sweep.py

report:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=reports reports/scaling_ensembles_report.tex

compile:
	$(BIN)/python -m compileall src

clean-cache:
	rm -rf outputs/*/cache outputs/series/*/cache

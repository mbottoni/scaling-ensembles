# Scaling Ensembles

Reusable PyTorch experiments for asking whether independently trained neural
networks become more functionally similar as parameter count grows.

The project is inspired by Fort, Hu, and Lakshminarayanan,
["Deep Ensembles: A Loss Landscape Perspective"](https://arxiv.org/abs/1912.02757).
The initial experiments measure function-space similarity between different SGD
minima across a width sweep.

## Install

```bash
cd scaling-ensembles
python -m pip install -e .
```

Install optional MLflow tracking support:

```bash
python -m pip install -e ".[tracking]"
```

## First Experiment

Run a small MNIST width sweep:

```bash
scaling-ensembles-sweep --config experiments/mnist_width_sweep.yaml
```

Run a CIFAR-10 convolutional width sweep:

```bash
scaling-ensembles-sweep --config experiments/cifar10_cnn_width_sweep.yaml
```

Run a CIFAR-10 patch-transformer width sweep:

```bash
scaling-ensembles-sweep --config experiments/cifar10_patch_transformer_width_sweep.yaml
```

This trains multiple random seeds for each width, saves checkpoints, computes
pairwise function similarity, and writes CSV summaries under the configured
output directory.

Sweeps are resumable by default. Existing checkpoints are reused, logits are
cached under each run's output directory, and `force_retrain: true` can be set in
the `cache` section when a clean rerun is needed.

Enable MLflow in any experiment YAML:

```yaml
tracking:
  mlflow_enabled: true
  tracking_uri: outputs/mlruns
  experiment_name: scaling-ensembles
  log_artifacts: true
  log_checkpoints: true
  log_logits: true
  log_environment: true
  use_mlflow_cache: true
```

When MLflow tracking is enabled, sweeps log the YAML config, environment and git
metadata, per-model metrics, pairwise/interpolation metrics, CSV summaries,
checkpoints, and cached logits. With `use_mlflow_cache: true`, a new run can
restore matching checkpoints and logits from previous completed MLflow runs when
the local files are missing.

On macOS, `device: auto` prefers PyTorch's Apple Metal backend (`mps`) before
checking CUDA. The CIFAR-10 configs set `device: mps` explicitly.

Generate paper-style plots from completed sweeps:

```bash
scaling-ensembles-paper-plots --output-root outputs
```

The plot command repeats several views from Fort et al.: diversity versus
accuracy, function similarity versus parameter count, interpolation barriers,
and final-minima weight/function similarity matrices. Points are labeled with
width and parameter count.

## Experiment Series

The `experiments/series/` directory contains larger follow-up configs for the
main research question:

> When does overparameterization collapse functional diversity between
> independently trained neural network minima?

The series includes extended width sweeps, matched train-loss controls, dataset
difficulty comparisons, and a longer patch-transformer schedule. See
`experiments/series/README.md` for the full list and rationale.

Open the Marimo app:

```bash
marimo edit notebooks/width_sweep.py
```

## Core Measurements

- Prediction agreement: fraction of inputs where two minima predict the same
  class.
- Error overlap: fraction of examples both models classify incorrectly.
- Mean Jensen-Shannon divergence: distance between predictive distributions.
- Mean logit cosine similarity: similarity between raw model outputs.
- Ensemble gain: improvement from averaging predictions.
- Linear loss barrier: loss increase along a straight parameter interpolation.

These metrics separate weight-space distance from function-space behavior, which
is the central object of the proposed paper.

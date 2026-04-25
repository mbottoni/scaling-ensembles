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

## First Experiment

Run a small MNIST width sweep:

```bash
scaling-ensembles-sweep --config experiments/mnist_width_sweep.yaml
```

This trains multiple random seeds for each width, saves checkpoints, computes
pairwise function similarity, and writes CSV summaries under the configured
output directory.

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

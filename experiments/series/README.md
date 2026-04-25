# Experiment Series

These configs are designed to study:

> When does overparameterization collapse functional diversity between independently trained neural network minima?

The series separates parameter count from major confounders: optimization quality,
dataset difficulty, architecture family, and transformer undertraining.

## 1. Clean Width Scaling

- `mnist_mlp_extended_width_sweep.yaml`
- `cifar10_cnn_extended_width_sweep.yaml`

These run larger width sweeps with more seeds than the pilot configs. Use these
to estimate how prediction agreement, JS divergence, ensemble gain, and linear
interpolation barriers scale with parameter count.

## 2. Matched Train-Loss Control

- `cifar10_cnn_matched_train_loss.yaml`

This config trains each width until it reaches a shared train-loss target, or
until `epochs` is exhausted. It tests whether apparent diversity collapse is
really a parameter-count effect or simply a consequence of larger models reaching
better endpoint solutions.

## 3. Dataset Difficulty

- `dataset_difficulty_mnist.yaml`
- `dataset_difficulty_fashionmnist.yaml`
- `dataset_difficulty_cifar10.yaml`

These use the same CNN width grid across increasingly difficult datasets. The
main hypothesis is that functional diversity should collapse more readily on
easy datasets and persist longer on harder datasets.

## 4. Transformer Training Control

- `cifar10_patch_transformer_long_schedule.yaml`

The pilot patch-transformer run likely undertrained. This longer schedule tests
whether the observed increase in diversity with width persists after improving
optimization.

## 5. Corrupted Evaluation

- `cifar10_cnn_gaussian_noise_eval.yaml`
- `cifar10_cnn_blur_eval.yaml`

These train on clean CIFAR-10 and evaluate pairwise function diversity on simple
corrupted test inputs. They are not a replacement for CIFAR-10-C, but they add
the first dataset-shift check needed to test whether functional diversity
collapses only on clean in-distribution examples.

## Running

Run one config:

```bash
scaling-ensembles-sweep --config experiments/series/cifar10_cnn_matched_train_loss.yaml
```

All sweeps support checkpoint and logits caching through the shared `cache`
section. This makes interrupted series runs resumable without retraining models
or recomputing logits:

```yaml
cache:
  enabled: true
  reuse_checkpoints: true
  reuse_logits: true
  force_retrain: false
```

MLflow can be enabled per config after installing `.[tracking]`:

```yaml
tracking:
  mlflow_enabled: true
  tracking_uri: outputs/mlruns
  experiment_name: scaling-ensembles-series
  log_artifacts: true
```

Run paper-style plots after a set of sweeps:

```bash
scaling-ensembles-paper-plots \
  --output-root outputs/series \
  --experiments cifar10-cnn-extended-width-sweep cifar10-cnn-matched-train-loss
```

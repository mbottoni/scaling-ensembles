#!/usr/bin/env python3
"""Evaluate clean-trained CIFAR-10 ensembles on the official CIFAR-10-C suite.

Reuses the existing CIFAR-10 SmallCNN checkpoints (no retraining) and measures,
per corruption type and severity, how individual accuracy, pairwise functional
diversity (disagreement), and 2-/M-member ensemble gain behave under
distribution shift.  The central test: do diversity and ensemble gain rise
monotonically with corruption severity, as the task-difficulty hypothesis
predicts?

Reads:
  data/CIFAR-10-C/*.npy                          (Hendrycks & Dietterich 2019)
  outputs/series/cifar10-cnn-extended-width-sweep/checkpoints/width_*_seed_*.pt
Writes:
  outputs/series/analysis/cifar10c_by_corruption.csv   (width x corruption x severity)
  outputs/series/analysis/cifar10c_by_severity.csv      (width x severity, mean over corruptions)
"""
from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from scaling_ensembles.config import load_config
from scaling_ensembles.data import DatasetInfo
from scaling_ensembles.train import load_checkpoint_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

CONFIG_PATH = "experiments/series/cifar10_cnn_extended_width_sweep.yaml"
CKPT_DIR = Path("outputs/series/cifar10-cnn-extended-width-sweep/checkpoints")
C_DIR = Path("data/CIFAR-10-C")
ANALYSIS = Path("outputs/series/analysis")

WIDTHS = [16, 32, 64, 128, 256]
SEEDS = [0, 1, 2, 3, 4]
SEVERITIES = [1, 2, 3, 4, 5]
BATCH = 1000

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# The 15 canonical CIFAR-10-C corruptions used for headline mCE-style reporting.
STANDARD = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
# Extra corruptions shipped with the dataset (reported but not in the canonical 15).
EXTRA = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess(images_uint8: np.ndarray) -> torch.Tensor:
    """(N,32,32,3) uint8 -> normalized (N,3,32,32) float tensor."""
    x = torch.from_numpy(images_uint8).float().div_(255.0).permute(0, 3, 1, 2)
    return (x - MEAN) / STD


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (preds, mean_softmax_probs) over batched inference."""
    preds, probs = [], []
    for i in range(0, x.shape[0], BATCH):
        logits = model(x[i : i + BATCH].to(device))
        p = logits.softmax(dim=1).cpu()
        probs.append(p)
        preds.append(p.argmax(dim=1))
    return torch.cat(preds), torch.cat(probs)


def ensemble_metrics(preds: list[torch.Tensor], probs: list[torch.Tensor], targets: torch.Tensor) -> dict:
    """Individual accuracy, pairwise disagreement, and ensemble gain.

    ``gain_pp`` is the 2-member gain averaged over all seed pairs (matching the
    rest of the paper's pairwise convention); ``gain_pp_m`` is the full
    M-member ensemble gain (all available seeds averaged).
    """
    accs = [(p == targets).float().mean().item() for p in preds]
    n = len(preds)
    pair_dis, pair_gain = [], []
    for i, j in itertools.combinations(range(n), 2):
        pair_dis.append((preds[i] != preds[j]).float().mean().item())
        ens_ij = ((probs[i] + probs[j]) / 2).argmax(dim=1)
        acc_ij = (ens_ij == targets).float().mean().item()
        pair_gain.append(acc_ij - 0.5 * (accs[i] + accs[j]))
    ens_probs = torch.stack(probs).mean(dim=0)
    ens_acc_m = (ens_probs.argmax(dim=1) == targets).float().mean().item()
    mean_indiv = float(np.mean(accs))
    return {
        "indiv_acc": mean_indiv,
        "disagreement": float(np.mean(pair_dis)),
        "gain_pp": float(np.mean(pair_gain)) * 100,       # 2-member (primary)
        "ensemble_acc_m": ens_acc_m,                       # M-member ensemble accuracy
        "gain_pp_m": (ens_acc_m - mean_indiv) * 100,       # M-member gain
    }


def clean_test() -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())
    x = torch.stack([img for img, _ in ds])  # (10000,3,32,32) in [0,1]
    x = (x - MEAN) / STD
    y = torch.tensor([lbl for _, lbl in ds])
    return x, y


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    LOGGER.info("Device: %s", device)
    config = load_config(CONFIG_PATH)
    info = DatasetInfo(input_shape=(3, 32, 32), num_classes=10)

    corruptions = [c for c in STANDARD + EXTRA if (C_DIR / f"{c}.npy").exists()]
    LOGGER.info("Found %d corruption files", len(corruptions))
    labels_np = np.load(C_DIR / "labels.npy")

    by_corruption, by_severity = [], []

    for width in WIDTHS:
        models = []
        for seed in SEEDS:
            ckpt = CKPT_DIR / f"width_{width}_seed_{seed}.pt"
            if not ckpt.exists():
                LOGGER.warning("missing %s", ckpt)
                continue
            models.append(load_checkpoint_model(ckpt, config, info, device))
        if len(models) < 2:
            LOGGER.warning("width %d: <2 models, skipping", width)
            continue
        n_models = len(models)

        # severity 0 = clean baseline
        cx, cy = clean_test()
        out = [predict(m, cx, device) for m in models]
        preds = [o[0] for o in out]
        probs = [o[1] for o in out]
        clean_m = ensemble_metrics(preds, probs, cy)
        by_severity.append({"width": width, "severity": 0, "n_models": n_models,
                            "n_corruptions": 0, **{k: round(v, 4) for k, v in clean_m.items()}})
        LOGGER.info("width %d sev 0 (clean): acc=%.3f disagree=%.3f gain=%.2fpp",
                    width, clean_m["indiv_acc"], clean_m["disagreement"], clean_m["gain_pp"])

        # accumulate per-severity means over the standard corruptions
        sev_accum: dict[int, list[dict]] = {s: [] for s in SEVERITIES}
        for corruption in corruptions:
            data = np.load(C_DIR / f"{corruption}.npy")  # (50000,32,32,3)
            for sev in SEVERITIES:
                sl = slice((sev - 1) * 10000, sev * 10000)
                x = preprocess(data[sl])
                y = torch.from_numpy(labels_np[sl]).long()
                out = [predict(mdl, x, device) for mdl in models]
                preds = [o[0] for o in out]
                probs = [o[1] for o in out]
                m = ensemble_metrics(preds, probs, y)
                by_corruption.append({
                    "width": width, "corruption": corruption, "severity": sev,
                    "standard": corruption in STANDARD,
                    **{k: round(v, 4) for k, v in m.items()},
                })
                if corruption in STANDARD:
                    sev_accum[sev].append(m)
            LOGGER.info("  width %d %s done", width, corruption)

        for sev in SEVERITIES:
            rows = sev_accum[sev]
            agg = {k: round(float(np.mean([r[k] for r in rows])), 4)
                   for k in ("indiv_acc", "disagreement", "gain_pp", "ensemble_acc_m", "gain_pp_m")}
            by_severity.append({"width": width, "severity": sev, "n_models": n_models,
                               "n_corruptions": len(rows), **agg})
            LOGGER.info("width %d sev %d (mean/%d std): acc=%.3f disagree=%.3f gain=%.2fpp",
                        width, sev, len(rows), agg["indiv_acc"], agg["disagreement"], agg["gain_pp"])

    _write(ANALYSIS / "cifar10c_by_corruption.csv", by_corruption)
    _write(ANALYSIS / "cifar10c_by_severity.csv", by_severity)


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        LOGGER.warning("no rows for %s", path)
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    LOGGER.info("Wrote %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()

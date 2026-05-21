from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class PairwiseSimilarity:
    width: int
    seed_a: int
    seed_b: int
    parameter_count: int
    agreement: float
    disagreement: float
    both_wrong: float
    error_jaccard: float
    js_divergence: float
    logit_cosine: float
    ensemble_accuracy: float
    ensemble_nll: float
    ensemble_brier: float
    model_a_accuracy: float
    model_b_accuracy: float
    model_a_nll: float
    model_b_nll: float
    model_a_brier: float
    model_b_brier: float


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    resolved_device = torch.device(device)
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for inputs, targets in loader:
        logits = model(inputs.to(resolved_device)).cpu()
        all_logits.append(logits)
        all_targets.append(targets.cpu())

    return torch.cat(all_logits), torch.cat(all_targets)


def compare_predictions(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets: torch.Tensor,
    *,
    width: int,
    seed_a: int,
    seed_b: int,
    parameter_count: int,
) -> PairwiseSimilarity:
    probs_a = logits_a.softmax(dim=1)
    probs_b = logits_b.softmax(dim=1)
    ensemble_probs = 0.5 * (probs_a + probs_b)
    preds_a = probs_a.argmax(dim=1)
    preds_b = probs_b.argmax(dim=1)
    ensemble_preds = ensemble_probs.argmax(dim=1)

    model_a_correct = preds_a == targets
    model_b_correct = preds_b == targets
    model_a_wrong = ~model_a_correct
    model_b_wrong = ~model_b_correct
    both_wrong = model_a_wrong & model_b_wrong
    either_wrong = model_a_wrong | model_b_wrong
    midpoint = ensemble_probs
    # Clamp to avoid log(0) which causes NaN in KL divergence
    probs_a_clamped = probs_a.clamp(min=1e-7)
    probs_b_clamped = probs_b.clamp(min=1e-7)
    midpoint_clamped = midpoint.clamp(min=1e-7)
    js = 0.5 * (
        F.kl_div(midpoint_clamped.log(), probs_a_clamped, reduction="batchmean")
        + F.kl_div(midpoint_clamped.log(), probs_b_clamped, reduction="batchmean")
    )
    agreement = (preds_a == preds_b).float().mean().item()

    return PairwiseSimilarity(
        width=width,
        seed_a=seed_a,
        seed_b=seed_b,
        parameter_count=parameter_count,
        agreement=agreement,
        disagreement=1.0 - agreement,
        both_wrong=both_wrong.float().mean().item(),
        error_jaccard=safe_divide(both_wrong.sum().item(), either_wrong.sum().item()),
        js_divergence=js.item(),
        logit_cosine=F.cosine_similarity(logits_a, logits_b, dim=1).mean().item(),
        ensemble_accuracy=(ensemble_preds == targets).float().mean().item(),
        ensemble_nll=negative_log_likelihood(ensemble_probs, targets),
        ensemble_brier=brier_score(ensemble_probs, targets),
        model_a_accuracy=model_a_correct.float().mean().item(),
        model_b_accuracy=model_b_correct.float().mean().item(),
        model_a_nll=negative_log_likelihood(probs_a, targets),
        model_b_nll=negative_log_likelihood(probs_b, targets),
        model_a_brier=brier_score(probs_a, targets),
        model_b_brier=brier_score(probs_b, targets),
    )


def negative_log_likelihood(probs: torch.Tensor, targets: torch.Tensor) -> float:
    selected = probs[torch.arange(targets.numel()), targets].clamp_min(1e-12)
    return (-selected.log()).mean().item()


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    one_hot = F.one_hot(targets, num_classes=probs.shape[1]).to(probs.dtype)
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) with equal-width bins."""
    confidences = probs.max(dim=1).values
    predictions = probs.argmax(dim=1)
    accuracies = (predictions == targets).float()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(targets.numel())
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_accuracy = accuracies[mask].mean().item()
        bin_confidence = confidences[mask].mean().item()
        ece += mask.sum().item() * abs(bin_accuracy - bin_confidence) / n
    return ece


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def write_pairwise_results(results: list[PairwiseSimilarity], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(PairwiseSimilarity.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in writer.fieldnames})

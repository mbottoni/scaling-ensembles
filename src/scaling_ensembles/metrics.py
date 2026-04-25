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
    both_wrong: float
    js_divergence: float
    logit_cosine: float
    ensemble_accuracy: float
    model_a_accuracy: float
    model_b_accuracy: float


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
    preds_a = probs_a.argmax(dim=1)
    preds_b = probs_b.argmax(dim=1)
    ensemble_preds = (probs_a + probs_b).argmax(dim=1)

    model_a_correct = preds_a == targets
    model_b_correct = preds_b == targets
    midpoint = 0.5 * (probs_a + probs_b)
    js = 0.5 * (
        F.kl_div(midpoint.log(), probs_a, reduction="batchmean")
        + F.kl_div(midpoint.log(), probs_b, reduction="batchmean")
    )

    return PairwiseSimilarity(
        width=width,
        seed_a=seed_a,
        seed_b=seed_b,
        parameter_count=parameter_count,
        agreement=(preds_a == preds_b).float().mean().item(),
        both_wrong=(~model_a_correct & ~model_b_correct).float().mean().item(),
        js_divergence=js.item(),
        logit_cosine=F.cosine_similarity(logits_a, logits_b, dim=1).mean().item(),
        ensemble_accuracy=(ensemble_preds == targets).float().mean().item(),
        model_a_accuracy=model_a_correct.float().mean().item(),
        model_b_accuracy=model_b_correct.float().mean().item(),
    )


def write_pairwise_results(results: list[PairwiseSimilarity], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(PairwiseSimilarity.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in writer.fieldnames})

"""Analyze singular-value energy of every pretrained ViT QK head operator."""

from __future__ import annotations

import torch

from experiments.qk_svd_oft.qk_svd import decompose_attention_qk
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import ViTForImageClassification


RANK_CANDIDATES = (4, 8, 12, 16, 24, 32, 40, 48, 56, 64)
ENERGY_THRESHOLDS = (0.90, 0.95, 0.99)


def percentile(values: torch.Tensor, quantile: float) -> float:
    return torch.quantile(values, quantile).item()


def summarize(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": values.min().item(),
        "p25": percentile(values, 0.25),
        "median": values.median().item(),
        "mean": values.mean().item(),
        "p75": percentile(values, 0.75),
        "max": values.max().item(),
    }


def print_energy_table(cumulative_energy: torch.Tensor) -> None:
    print("\nCumulative Frobenius-energy ratio across 144 heads")
    print(
        "rank |    min |    p25 | median |   mean |    p75 |    max | "
        ">=90% | >=95% | >=99%"
    )
    print("-" * 96)
    for rank in RANK_CANDIDATES:
        values = cumulative_energy[:, rank - 1]
        stats = summarize(values)
        coverage = [
            100.0 * (values >= threshold).float().mean().item()
            for threshold in ENERGY_THRESHOLDS
        ]
        print(
            f"{rank:4d} | {stats['min']:6.3f} | {stats['p25']:6.3f} | "
            f"{stats['median']:6.3f} | {stats['mean']:6.3f} | "
            f"{stats['p75']:6.3f} | {stats['max']:6.3f} | "
            f"{coverage[0]:5.1f}% | {coverage[1]:5.1f}% | {coverage[2]:5.1f}%"
        )


def print_threshold_table(cumulative_energy: torch.Tensor) -> None:
    print("\nSmallest per-head rank reaching each energy threshold")
    print("target | min | p25 | median | mean | p75 | max")
    print("-" * 54)
    for threshold in ENERGY_THRESHOLDS:
        required_rank = (cumulative_energy < threshold).sum(dim=-1) + 1
        stats = summarize(required_rank.to(torch.float64))
        print(
            f"{threshold:5.0%} | {stats['min']:3.0f} | {stats['p25']:4.1f} | "
            f"{stats['median']:6.1f} | {stats['mean']:4.1f} | "
            f"{stats['p75']:4.1f} | {stats['max']:3.0f}"
        )


def print_layer_table(cumulative_energy: torch.Tensor) -> None:
    layer_energy = cumulative_energy.reshape(12, 12, 64).mean(dim=1)
    displayed_ranks = (16, 24, 32, 40, 48, 56)
    print("\nMean cumulative energy within each Transformer layer")
    print("layer | " + " | ".join(f"r={rank:2d}" for rank in displayed_ranks))
    print("-" * 59)
    for layer_index, values in enumerate(layer_energy, start=1):
        entries = " | ".join(
            f"{values[rank - 1].item():5.3f}" for rank in displayed_ranks
        )
        print(f"{layer_index:5d} | {entries}")


def print_rank_shape_statistics(singular_values: torch.Tensor) -> None:
    squared = singular_values.square()
    stable_rank = squared.sum(dim=-1) / squared[:, 0]
    probabilities = squared / squared.sum(dim=-1, keepdim=True)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    )

    stable = summarize(stable_rank)
    effective = summarize(effective_rank)
    print("\nSpectrum-shape diagnostics across 144 heads")
    print("metric         | min  | p25  | median | mean | p75  | max")
    print("-" * 65)
    print(
        f"stable rank    | {stable['min']:4.1f} | {stable['p25']:4.1f} | "
        f"{stable['median']:6.1f} | {stable['mean']:4.1f} | "
        f"{stable['p75']:4.1f} | {stable['max']:4.1f}"
    )
    print(
        f"effective rank | {effective['min']:4.1f} | {effective['p25']:4.1f} | "
        f"{effective['median']:6.1f} | {effective['mean']:4.1f} | "
        f"{effective['p75']:4.1f} | {effective['max']:4.1f}"
    )


def main() -> None:
    model = ViTForImageClassification(num_classes=3)
    loaded = load_pretrained_vit(model)
    if loaded != 198:
        raise RuntimeError(f"Expected 198 pretrained tensors, loaded {loaded}.")
    model.eval()

    spectra = []
    with torch.no_grad():
        for block in model.encoder.blocks:
            decomposition = decompose_attention_qk(block.attention)
            spectra.append(decomposition.singular_values.to(torch.float64))

    singular_values = torch.cat(spectra, dim=0)
    squared = singular_values.square()
    cumulative_energy = squared.cumsum(dim=-1) / squared.sum(
        dim=-1,
        keepdim=True,
    )

    print(f"Analyzed singular-value matrix: {tuple(singular_values.shape)}")
    print_energy_table(cumulative_energy)
    print_threshold_table(cumulative_energy)
    print_layer_table(cumulative_energy)
    print_rank_shape_statistics(singular_values)


if __name__ == "__main__":
    main()

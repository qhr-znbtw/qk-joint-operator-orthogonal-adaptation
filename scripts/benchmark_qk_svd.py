"""Benchmark compact QK SVD on one block and the complete ViT-B/16."""

from __future__ import annotations

import statistics
import time

import torch

from experiments.qk_svd_oft.qk_svd import decompose_attention_qk
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import ViTForImageClassification


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_blocks(
    blocks: list[torch.nn.Module],
    device: torch.device,
    repeats: int = 5,
) -> float:
    timings = []
    with torch.no_grad():
        for _ in range(2):
            for block in blocks:
                decompose_attention_qk(block.attention)
        synchronize(device)

        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            for block in blocks:
                decompose_attention_qk(block.attention)
            synchronize(device)
            timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def run_device(model: ViTForImageClassification, device: torch.device) -> None:
    model.to(device)
    blocks = list(model.encoder.blocks)
    one_block_seconds = benchmark_blocks(blocks[:1], device)
    all_blocks_seconds = benchmark_blocks(blocks, device)
    print(
        f"{device.type.upper()}: 12 heads={one_block_seconds * 1000:.2f} ms, "
        f"144 heads={all_blocks_seconds * 1000:.2f} ms"
    )


def main() -> None:
    model = ViTForImageClassification(num_classes=3)
    loaded = load_pretrained_vit(model)
    if loaded != 198:
        raise RuntimeError(f"Expected 198 pretrained tensors, loaded {loaded}.")
    model.eval()

    run_device(model, torch.device("cpu"))
    if torch.cuda.is_available():
        run_device(model, torch.device("cuda"))


if __name__ == "__main__":
    main()

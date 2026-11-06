"""Shared training utilities for full-spectrum joint-QK BOFT variants."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiments.qk_boft_left.config import (
    BOFT_BLOCK_SIZE,
    BOFT_LEARNING_RATE,
    BOFT_NUM_FACTORS,
    CLASSIFIER_LEARNING_RATE,
    NUM_EPOCHS,
    QK_SPECTRAL_RANK,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from vit_core.evaluation import test
from vit_core.mini_vtab import (
    EVAL_BATCH_SIZE,
    RANDOM_SEED,
    TASK_SPECS,
    TRAIN_BATCH_SIZE,
    build_dataloaders,
    get_task_spec,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.training_engine import train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--boft-lr", type=float, default=BOFT_LEARNING_RATE)
    parser.add_argument("--external-boft-lr", type=float, default=1e-4)
    parser.add_argument("--spectrum-lr", type=float, default=1e-4)
    parser.add_argument(
        "--classifier-lr", type=float, default=CLASSIFIER_LEARNING_RATE
    )
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    *,
    boft_lr: float = BOFT_LEARNING_RATE,
    external_boft_lr: float | None = None,
    spectrum_lr: float | None = None,
    classifier_lr: float = CLASSIFIER_LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> AdamW:
    boft_parameters = []
    external_boft_parameters = []
    spectrum_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("external_boft.") and name.endswith("skew_parameters"):
            external_boft_parameters.append(parameter)
        elif name.endswith("skew_parameters"):
            boft_parameters.append(parameter)
        elif name.endswith("spectrum_log_scales"):
            spectrum_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
    if not boft_parameters or not classifier_parameters:
        raise RuntimeError("BOFT and classifier groups must both be non-empty.")
    if spectrum_parameters and spectrum_lr is None:
        raise RuntimeError("spectrum_lr is required for learnable spectrum parameters.")
    parameter_groups = [{"params": boft_parameters, "lr": boft_lr}]
    if external_boft_parameters:
        if external_boft_lr is None:
            raise RuntimeError(
                "external_boft_lr is required for external BOFT parameters."
            )
        parameter_groups.append(
            {"params": external_boft_parameters, "lr": external_boft_lr}
        )
    if spectrum_parameters:
        parameter_groups.append({"params": spectrum_parameters, "lr": spectrum_lr})
    parameter_groups.append(
        {"params": classifier_parameters, "lr": classifier_lr}
    )
    return AdamW(parameter_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: AdamW,
    *,
    num_epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    if not 0 <= warmup_epochs < num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs).")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = num_epochs - warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(decay_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def count_parameters(
    model: nn.Module,
    expected_boft: int,
    expected_spectrum: int = 0,
) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    boft = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("skew_parameters")
    )
    spectrum = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("spectrum_log_scales")
    )
    classifier = sum(parameter.numel() for parameter in model.head.classifier.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if boft != expected_boft:
        raise RuntimeError(f"Expected {expected_boft:,} BOFT parameters, got {boft:,}.")
    if spectrum != expected_spectrum:
        raise RuntimeError(
            f"Expected {expected_spectrum:,} spectrum parameters, got {spectrum:,}."
        )
    if trainable != boft + spectrum + classifier:
        raise RuntimeError(
            "Trainable count must equal BOFT plus spectrum plus classifier."
        )
    return {
        "total": total,
        "trainable": trainable,
        "boft": boft,
        "spectrum": spectrum,
        "classifier": classifier,
    }


def _parameter_norm(parameters: list[torch.Tensor], *, gradient: bool) -> float:
    terms = []
    for parameter in parameters:
        value = parameter.grad if gradient else parameter
        if value is not None:
            terms.append(value.float().square().sum())
    if not terms:
        return 0.0
    return float(torch.stack(terms).sum().sqrt().item())


def _rotation_distances(rotations: torch.Tensor) -> tuple[float, float]:
    identity = torch.eye(
        rotations.shape[-1], dtype=rotations.dtype, device=rotations.device
    )
    distances = torch.linalg.matrix_norm(
        rotations.float() - identity, ord="fro", dim=(-2, -1)
    ) / math.sqrt(rotations.shape[-1])
    return float(distances.mean().item()), float(distances.max().item())


@torch.no_grad()
def single_side_diagnostics(model: nn.Module) -> dict[str, float]:
    layers = model.adapter_layers()
    parameters = [layer.skew_parameters for layer in layers]
    blocks = torch.cat([layer.orthogonal_blocks().float() for layer in layers])
    dense = torch.cat([layer.dense_orthogonal_matrices().float() for layer in layers])
    block_mean, block_max = _rotation_distances(blocks)
    dense_mean, dense_max = _rotation_distances(dense)
    return {
        "rotation_gradient_norm": _parameter_norm(parameters, gradient=True),
        "rotation_parameter_norm": _parameter_norm(parameters, gradient=False),
        "mean_block_rotation_distance": block_mean,
        "max_block_rotation_distance": block_max,
        "mean_rotation_distance": dense_mean,
        "max_rotation_distance": dense_max,
    }


@torch.no_grad()
def bilateral_diagnostics(model: nn.Module) -> dict[str, float]:
    layers = model.adapter_layers()
    output: dict[str, float] = {}
    for side in ("left", "right"):
        parameters = [getattr(layer, f"{side}_skew_parameters") for layer in layers]
        blocks = torch.cat(
            [getattr(layer, f"{side}_orthogonal_blocks")().float() for layer in layers]
        )
        dense = torch.cat(
            [getattr(layer, f"dense_{side}_orthogonal_matrices")().float() for layer in layers]
        )
        block_mean, block_max = _rotation_distances(blocks)
        dense_mean, dense_max = _rotation_distances(dense)
        output.update(
            {
                f"{side}_rotation_gradient_norm": _parameter_norm(parameters, gradient=True),
                f"{side}_rotation_parameter_norm": _parameter_norm(parameters, gradient=False),
                f"{side}_mean_block_rotation_distance": block_mean,
                f"{side}_max_block_rotation_distance": block_max,
                f"{side}_mean_rotation_distance": dense_mean,
                f"{side}_max_rotation_distance": dense_max,
            }
        )
    return output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(
    *,
    args: argparse.Namespace,
    model_factory: Callable[[int], nn.Module],
    method: str,
    label: str,
    side: str,
    expected_boft: int,
    diagnostics_fn: Callable[[nn.Module], dict[str, float]],
    expected_spectrum: int = 0,
    spectrum_lr: float | None = None,
    spectrum_parameterization: str | None = None,
    external_boft_lr: float | None = None,
    extra_result_metadata: dict[str, object] | None = None,
) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    task = get_task_spec(args.task)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / method / "development_800_200" / task.name / f"seed_{args.seed}.pt"
    result_path = PROJECT_ROOT / "results" / method / "development_800_200" / task.name / f"seed_{args.seed}.json"

    train_loader, val_loader, test_loader = build_dataloaders(
        task.name,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    model = model_factory(task.num_classes)
    loaded_tensors = load_pretrained_vit(model)
    model.initialize_adapters()
    optimizer = build_optimizer(
        model,
        boft_lr=args.boft_lr,
        external_boft_lr=external_boft_lr,
        spectrum_lr=spectrum_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )
    criterion = nn.CrossEntropyLoss()
    counts = count_parameters(model, expected_boft, expected_spectrum)

    print(f"Task: {task.name} ({task.group}, {task.num_classes} classes)")
    print(f"Device: {device}")
    print(f"Dataset sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Method: {label}, rank=64, block={BOFT_BLOCK_SIZE}, factors={BOFT_NUM_FACTORS}")
    print(f"Loaded backbone tensors: {loaded_tensors}")
    print(
        f"Parameters: {counts['trainable']:,} trainable / {counts['total']:,} "
        f"total; BOFT={counts['boft']:,}; spectrum={counts['spectrum']:,}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_start = time.perf_counter()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        epoch_metrics_fn=diagnostics_fn,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start
    test_start = time.perf_counter()
    test_metrics = test(model, test_loader, criterion, device)
    synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))

    result = {
        "protocol": "development_800_200",
        "method": method,
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "operator": "bias_augmented_per_head_qk",
        "side": side,
        "rank": QK_SPECTRAL_RANK,
        "num_heads_adapted": 144,
        "spectrum": "full_compact",
        "singular_values_trainable": expected_spectrum > 0,
        "spectrum_parameterization": spectrum_parameterization,
        "rotation": "block_butterfly_scaled_cayley",
        "rotation_initialization": "identity",
        "boft_block_size": BOFT_BLOCK_SIZE,
        "boft_num_factors": BOFT_NUM_FACTORS,
        "boft_learning_rate": args.boft_lr,
        "external_boft_learning_rate": external_boft_lr,
        "spectrum_learning_rate": spectrum_lr,
        "classifier_learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "dataset_sizes": {"train": len(train_loader.dataset), "validation": len(val_loader.dataset), "test": len(test_loader.dataset)},
        "loaded_backbone_tensors": loaded_tensors,
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "boft_parameters": counts["boft"],
        "spectrum_parameters": counts["spectrum"],
        "classifier_parameters": counts["classifier"],
        "trainable_fraction": counts["trainable"] / counts["total"],
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "best_epoch": int(best_epoch["epoch"]),
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "test": test_metrics,
        "history": history,
    }
    if extra_result_metadata:
        duplicate_keys = set(result) & set(extra_result_metadata)
        if duplicate_keys:
            raise KeyError(
                f"Extra result metadata duplicates existing keys: {duplicate_keys}"
            )
        result.update(extra_result_metadata)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\nFinal test result")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Peak VRAM: {peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"Result file: {result_path}")

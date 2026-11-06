"""Train full-spectrum left-sided joint-QK BOFT on Mini-VTAB-3."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

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
from experiments.qk_boft_left.model import ViTQKBOFTLeftClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--boft-lr", type=float, default=BOFT_LEARNING_RATE)
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=CLASSIFIER_LEARNING_RATE,
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
    classifier_lr: float = CLASSIFIER_LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> AdamW:
    boft_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".skew_parameters"):
            boft_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
    if not boft_parameters or not classifier_parameters:
        raise RuntimeError("BOFT and classifier groups must both be non-empty.")
    return AdamW(
        [
            {"params": boft_parameters, "lr": boft_lr},
            {"params": classifier_parameters, "lr": classifier_lr},
        ],
        weight_decay=weight_decay,
    )


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


def count_parameters(model: ViTQKBOFTLeftClassifier) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    boft = sum(layer.skew_parameters.numel() for layer in model.adapter_layers())
    classifier = sum(parameter.numel() for parameter in model.head.classifier.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if boft != 69_120:
        raise RuntimeError(f"Expected 69,120 BOFT parameters, got {boft:,}.")
    if trainable != boft + classifier:
        raise RuntimeError("Trainable count must equal BOFT plus classifier.")
    return {
        "total": total,
        "trainable": trainable,
        "boft": boft,
        "classifier": classifier,
    }


@torch.no_grad()
def rotation_diagnostics(model: nn.Module) -> dict[str, float]:
    """Measure block and composed rotation scales after each epoch."""
    if not isinstance(model, ViTQKBOFTLeftClassifier):
        raise TypeError("Expected ViTQKBOFTLeftClassifier.")
    layers = model.adapter_layers()
    parameter_squared_norm = sum(
        layer.skew_parameters.float().square().sum() for layer in layers
    )
    gradient_squared_norm = sum(
        layer.skew_parameters.grad.float().square().sum()
        for layer in layers
        if layer.skew_parameters.grad is not None
    )
    blocks = torch.cat(
        [layer.orthogonal_blocks().float() for layer in layers],
        dim=0,
    )
    block_identity = torch.eye(
        blocks.shape[-1],
        dtype=blocks.dtype,
        device=blocks.device,
    )
    block_distances = torch.linalg.matrix_norm(
        blocks - block_identity,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(blocks.shape[-1])
    dense_rotations = torch.cat(
        [layer.dense_orthogonal_matrices().float() for layer in layers],
        dim=0,
    )
    dense_identity = torch.eye(
        dense_rotations.shape[-1],
        dtype=dense_rotations.dtype,
        device=dense_rotations.device,
    )
    dense_distances = torch.linalg.matrix_norm(
        dense_rotations - dense_identity,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(dense_rotations.shape[-1])
    return {
        "rotation_gradient_norm": float(gradient_squared_norm.sqrt().item()),
        "rotation_parameter_norm": float(parameter_squared_norm.sqrt().item()),
        "mean_block_rotation_distance": float(block_distances.mean().item()),
        "max_block_rotation_distance": float(block_distances.max().item()),
        "mean_rotation_distance": float(dense_distances.mean().item()),
        "max_rotation_distance": float(dense_distances.max().item()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    task = get_task_spec(args.task)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "qk_boft_left_full"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.pt"
    )
    result_path = (
        PROJECT_ROOT
        / "results"
        / "qk_boft_left_full"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.json"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        task.name,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    model = ViTQKBOFTLeftClassifier(num_classes=task.num_classes)
    loaded_tensors = load_pretrained_vit(model)
    model.initialize_adapters()
    optimizer = build_optimizer(
        model,
        boft_lr=args.boft_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    counts = count_parameters(model)

    print(f"Task: {task.name} ({task.group}, {task.num_classes} classes)")
    print(f"Device: {device}")
    print(
        "Dataset sizes: "
        f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/"
        f"{len(test_loader.dataset)}"
    )
    print(
        "Method: QK-BOFT-Full-Left, rank=64, "
        f"block={BOFT_BLOCK_SIZE}, factors={BOFT_NUM_FACTORS}"
    )
    print(f"Loaded backbone tensors: {loaded_tensors}")
    print(
        f"Parameters: {counts['trainable']:,} trainable / "
        f"{counts['total']:,} total; BOFT={counts['boft']:,}"
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
        epoch_metrics_fn=rotation_diagnostics,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    test_start = time.perf_counter()
    test_metrics = test(model, test_loader, criterion, device)
    synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))

    result = {
        "protocol": "development_800_200",
        "method": "qk_boft_left_full",
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "operator": "bias_augmented_per_head_qk",
        "side": "left",
        "rank": QK_SPECTRAL_RANK,
        "num_heads_adapted": 144,
        "spectrum": "full_compact",
        "singular_values_trainable": False,
        "rotation": "block_butterfly_scaled_cayley",
        "rotation_initialization": "identity",
        "boft_block_size": BOFT_BLOCK_SIZE,
        "boft_num_factors": BOFT_NUM_FACTORS,
        "boft_learning_rate": args.boft_lr,
        "classifier_learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "loaded_backbone_tensors": loaded_tensors,
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "boft_parameters": counts["boft"],
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
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\nFinal test result")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Peak VRAM: {peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"Result file: {result_path}")


if __name__ == "__main__":
    main()


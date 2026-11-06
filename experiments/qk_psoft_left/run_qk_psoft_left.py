"""Train full-spectrum left-sided joint-QK PSOFT on Mini-VTAB-3."""

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

from experiments.qk_psoft_left.attention import QKPSOFTLeftAttention
from experiments.qk_psoft_left.config import (
    CLASSIFIER_LEARNING_RATE,
    NUM_EPOCHS,
    QK_PSOFT_RANK,
    ROTATION_LEARNING_RATE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from experiments.qk_psoft_left.model import ViTQKPSOFTLeftClassifier
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
    parser.add_argument("--rotation-lr", type=float, default=ROTATION_LEARNING_RATE)
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
    rotation_lr: float = ROTATION_LEARNING_RATE,
    classifier_lr: float = CLASSIFIER_LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> AdamW:
    rotation_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".skew_parameters"):
            rotation_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
    if not rotation_parameters or not classifier_parameters:
        raise RuntimeError("Rotation and classifier groups must both be non-empty.")
    return AdamW(
        [
            {"params": rotation_parameters, "lr": rotation_lr},
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


def count_parameters(model: ViTQKPSOFTLeftClassifier) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    rotation = sum(layer.skew_parameters.numel() for layer in model.adapter_layers())
    classifier = sum(parameter.numel() for parameter in model.head.classifier.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if rotation != 290_304:
        raise RuntimeError(f"Expected 290,304 rotation parameters, got {rotation:,}.")
    if trainable != rotation + classifier:
        raise RuntimeError("Trainable count must equal rotations plus classifier.")
    return {
        "total": total,
        "trainable": trainable,
        "rotation": rotation,
        "classifier": classifier,
    }


@torch.no_grad()
def rotation_diagnostics(model: nn.Module) -> dict[str, float]:
    """Measure optimization scale without changing the training state."""
    if not isinstance(model, ViTQKPSOFTLeftClassifier):
        raise TypeError("Expected ViTQKPSOFTLeftClassifier.")
    layers = model.adapter_layers()
    parameter_squared_norm = sum(
        layer.skew_parameters.float().square().sum() for layer in layers
    )
    gradient_squared_norm = sum(
        layer.skew_parameters.grad.float().square().sum()
        for layer in layers
        if layer.skew_parameters.grad is not None
    )
    rotations = torch.cat(
        [layer.orthogonal_matrices().float() for layer in layers],
        dim=0,
    )
    identity = torch.eye(
        rotations.shape[-1],
        device=rotations.device,
        dtype=rotations.dtype,
    )
    normalized_distances = torch.linalg.matrix_norm(
        rotations - identity,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(rotations.shape[-1])
    return {
        "rotation_gradient_norm": float(gradient_squared_norm.sqrt().item()),
        "rotation_parameter_norm": float(parameter_squared_norm.sqrt().item()),
        "mean_rotation_distance": float(normalized_distances.mean().item()),
        "max_rotation_distance": float(normalized_distances.max().item()),
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
        / "qk_psoft_left_full"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.pt"
    )
    result_path = (
        PROJECT_ROOT
        / "results"
        / "qk_psoft_left_full"
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
    model = ViTQKPSOFTLeftClassifier(num_classes=task.num_classes)
    loaded_tensors = load_pretrained_vit(model)
    model.initialize_adapters()
    optimizer = build_optimizer(
        model,
        rotation_lr=args.rotation_lr,
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
    print("Method: QK-PSOFT-Full-Left, rank=64, 144 dense rotations")
    print(f"Loaded backbone tensors: {loaded_tensors}")
    print(
        f"Parameters: {counts['trainable']:,} trainable / "
        f"{counts['total']:,} total; rotations={counts['rotation']:,}"
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
        "method": "qk_psoft_left_full",
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "operator": "bias_augmented_per_head_qk",
        "side": "left",
        "rank": QK_PSOFT_RANK,
        "num_heads_adapted": 144,
        "spectrum": "full_compact",
        "singular_values_trainable": False,
        "rotation": "dense_scaled_cayley",
        "rotation_initialization": "identity",
        "rotation_learning_rate": args.rotation_lr,
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
        "rotation_parameters": counts["rotation"],
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

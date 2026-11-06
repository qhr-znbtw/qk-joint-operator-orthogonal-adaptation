"""Train full-spectrum bilateral joint-QK PSOFT on Mini-VTAB-3."""

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

from experiments.qk_psoft_bilateral.config import (
    CLASSIFIER_LEARNING_RATE,
    NUM_EPOCHS,
    QK_PSOFT_RANK,
    ROTATION_LEARNING_RATE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from experiments.qk_psoft_bilateral.model import ViTQKPSOFTBilateralClassifier
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
        if name.endswith("_skew_parameters"):
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


def count_parameters(model: ViTQKPSOFTBilateralClassifier) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    left_rotation = sum(
        layer.left_skew_parameters.numel() for layer in model.adapter_layers()
    )
    right_rotation = sum(
        layer.right_skew_parameters.numel() for layer in model.adapter_layers()
    )
    rotation = left_rotation + right_rotation
    classifier = sum(parameter.numel() for parameter in model.head.classifier.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if left_rotation != 290_304 or right_rotation != 290_304:
        raise RuntimeError(
            "Expected 290,304 parameters on each rotation side, got "
            f"{left_rotation:,} and {right_rotation:,}."
        )
    if trainable != rotation + classifier:
        raise RuntimeError("Trainable count must equal rotations plus classifier.")
    return {
        "total": total,
        "trainable": trainable,
        "rotation": rotation,
        "left_rotation": left_rotation,
        "right_rotation": right_rotation,
        "classifier": classifier,
    }


def _squared_gradient_norm(parameters: list[torch.Tensor]) -> torch.Tensor:
    terms = [
        parameter.grad.float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not terms:
        return parameters[0].new_zeros((), dtype=torch.float32)
    return sum(terms)


@torch.no_grad()
def rotation_diagnostics(model: nn.Module) -> dict[str, float]:
    """Measure the two rotation sides independently after each epoch."""
    if not isinstance(model, ViTQKPSOFTBilateralClassifier):
        raise TypeError("Expected ViTQKPSOFTBilateralClassifier.")
    layers = model.adapter_layers()
    left_parameters = [layer.left_skew_parameters for layer in layers]
    right_parameters = [layer.right_skew_parameters for layer in layers]
    left_parameter_squared_norm = sum(
        parameter.float().square().sum() for parameter in left_parameters
    )
    right_parameter_squared_norm = sum(
        parameter.float().square().sum() for parameter in right_parameters
    )
    left_gradient_squared_norm = _squared_gradient_norm(left_parameters)
    right_gradient_squared_norm = _squared_gradient_norm(right_parameters)
    left_rotations = torch.cat(
        [layer.left_orthogonal_matrices().float() for layer in layers],
        dim=0,
    )
    right_rotations = torch.cat(
        [layer.right_orthogonal_matrices().float() for layer in layers],
        dim=0,
    )
    identity = torch.eye(
        left_rotations.shape[-1],
        device=left_rotations.device,
        dtype=left_rotations.dtype,
    )
    left_distances = torch.linalg.matrix_norm(
        left_rotations - identity,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(left_rotations.shape[-1])
    right_distances = torch.linalg.matrix_norm(
        right_rotations - identity,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(right_rotations.shape[-1])
    all_distances = torch.cat((left_distances, right_distances))
    return {
        "rotation_gradient_norm": float(
            (left_gradient_squared_norm + right_gradient_squared_norm).sqrt().item()
        ),
        "left_rotation_gradient_norm": float(
            left_gradient_squared_norm.sqrt().item()
        ),
        "right_rotation_gradient_norm": float(
            right_gradient_squared_norm.sqrt().item()
        ),
        "rotation_parameter_norm": float(
            (
                left_parameter_squared_norm + right_parameter_squared_norm
            ).sqrt().item()
        ),
        "left_rotation_parameter_norm": float(
            left_parameter_squared_norm.sqrt().item()
        ),
        "right_rotation_parameter_norm": float(
            right_parameter_squared_norm.sqrt().item()
        ),
        "mean_rotation_distance": float(all_distances.mean().item()),
        "max_rotation_distance": float(all_distances.max().item()),
        "left_mean_rotation_distance": float(left_distances.mean().item()),
        "left_max_rotation_distance": float(left_distances.max().item()),
        "right_mean_rotation_distance": float(right_distances.mean().item()),
        "right_max_rotation_distance": float(right_distances.max().item()),
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
        / "qk_psoft_bilateral_full"
        / "development_800_200"
        / task.name
        / f"seed_{args.seed}.pt"
    )
    result_path = (
        PROJECT_ROOT
        / "results"
        / "qk_psoft_bilateral_full"
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
    model = ViTQKPSOFTBilateralClassifier(num_classes=task.num_classes)
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
    print("Method: QK-PSOFT-Full-Bilateral, rank=64, 288 dense rotations")
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
        "method": "qk_psoft_bilateral_full",
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "operator": "bias_augmented_per_head_qk",
        "side": "bilateral",
        "rank": QK_PSOFT_RANK,
        "num_heads_adapted": 144,
        "rotations_per_head": 2,
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
        "left_rotation_parameters": counts["left_rotation"],
        "right_rotation_parameters": counts["right_rotation"],
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


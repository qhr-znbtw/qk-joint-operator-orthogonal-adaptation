"""Training loops shared by all transfer-learning experiments."""

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from vit_core.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from vit_core.evaluation import test


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch and return sample-weighted loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training DataLoader is empty.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    checkpoint_path: str | Path,
    scheduler: LRScheduler | None = None,
    epoch_metrics_fn: Callable[[nn.Module], dict[str, float]] | None = None,
) -> list[dict[str, float | int]]:
    """Train fixed epochs and restore the best validation checkpoint."""
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0

    for epoch in range(1, num_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics = test(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        if epoch_metrics_fn is not None:
            extra_metrics = epoch_metrics_fn(model)
            duplicate_keys = set(epoch_metrics) & set(extra_metrics)
            if duplicate_keys:
                raise KeyError(
                    f"Extra epoch metrics duplicate existing keys: {duplicate_keys}"
                )
            epoch_metrics.update(extra_metrics)
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"lr {learning_rate:.2e} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"train acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, "
            f"val acc {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            save_trainable_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                val_accuracy=best_val_accuracy,
            )
        if scheduler is not None:
            scheduler.step()

    load_trainable_checkpoint(model, checkpoint_path)
    model.to(device)
    return history

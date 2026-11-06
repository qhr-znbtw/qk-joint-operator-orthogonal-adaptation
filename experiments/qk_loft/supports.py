"""Build fixed Principal, GradSVD, or SkewGrad supports for joint QK."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import nn

from experiments.qk_loft.attention import QKCentralCalibrationAttention
from vit_core.vit_backbone import ViTForImageClassification


def _row_orthonormalize(rows: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal row basis with the same batched row span."""
    columns, _ = torch.linalg.qr(rows.transpose(-1, -2), mode="reduced")
    return columns.transpose(-1, -2).contiguous()


def _check_support(support: torch.Tensor, *, rank: int, width: int) -> None:
    expected = (support.shape[0], rank, width)
    if support.shape != expected:
        raise RuntimeError(f"Expected support {expected}, got {tuple(support.shape)}.")
    gram = support.float() @ support.float().transpose(-1, -2)
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype).expand_as(gram)
    if not torch.allclose(gram, identity, atol=3e-4, rtol=3e-4):
        raise RuntimeError("Constructed QK-LOFT support is not row-orthonormal.")


def principal_support(
    num_heads: int,
    width: int,
    rank: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Select the first ``rank`` compact singular coordinates."""
    if not 0 < rank <= width:
        raise ValueError("rank must be in [1, width].")
    support = torch.eye(width, device=device, dtype=torch.float32)[:rank]
    support = support.unsqueeze(0).expand(num_heads, -1, -1).clone()
    _check_support(support, rank=rank, width=width)
    return support


def supports_from_gradient(
    gradient: torch.Tensor,
    singular_values: torch.Tensor,
    *,
    support_type: str,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive left and right supports from the compact-core gradient.

    For ``B = diag(singular_values)``, GradSVD uses the leading left and
    right singular spaces of ``dL/dB``. SkewGrad instead uses the dominant
    invariant spaces of ``skew(G B^T)`` and ``skew(B^T G)``.
    """
    if support_type not in {"grad_svd", "skew_grad"}:
        raise ValueError("support_type must be 'grad_svd' or 'skew_grad'.")
    if gradient.ndim != 3 or gradient.shape[-1] != gradient.shape[-2]:
        raise ValueError("gradient must have shape [heads, width, width].")
    if singular_values.shape != gradient.shape[:2]:
        raise ValueError("singular_values must have shape [heads, width].")
    width = gradient.shape[-1]
    if not 0 < rank <= width:
        raise ValueError("rank must be in [1, width].")
    if support_type == "skew_grad" and rank % 2:
        raise ValueError("SkewGrad rank must be even to preserve skew pairs.")

    gradient = gradient.float()
    if support_type == "grad_svd":
        left_vectors, _, right_vectors_transpose = torch.linalg.svd(
            gradient,
            full_matrices=False,
        )
        left = left_vectors[..., :rank].transpose(-1, -2)
        right = right_vectors_transpose[..., :rank, :]
    else:
        base = torch.diag_embed(singular_values.float())
        left_cross = gradient @ base.transpose(-1, -2)
        right_cross = base.transpose(-1, -2) @ gradient
        left_signal = 0.5 * (left_cross - left_cross.transpose(-1, -2))
        right_signal = 0.5 * (right_cross - right_cross.transpose(-1, -2))
        _, _, left_right_t = torch.linalg.svd(left_signal, full_matrices=False)
        _, _, right_right_t = torch.linalg.svd(right_signal, full_matrices=False)
        left = left_right_t[..., :rank, :]
        right = right_right_t[..., :rank, :]

    left = _row_orthonormalize(left).contiguous()
    right = _row_orthonormalize(right).contiguous()
    _check_support(left, rank=rank, width=width)
    _check_support(right, rank=rank, width=width)
    return left, right


def build_principal_supports(
    model: ViTForImageClassification,
    *,
    rank: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    """Build the canonical principal-coordinate support for every layer."""
    start = time.perf_counter()
    left_supports: list[torch.Tensor] = []
    right_supports: list[torch.Tensor] = []
    for block in model.encoder.blocks:
        attention = block.attention
        support = principal_support(
            attention.num_heads,
            attention.head_size,
            rank,
        )
        left_supports.append(support)
        right_supports.append(support.clone())
    return left_supports, right_supports, time.perf_counter() - start


def build_gradient_supports(
    model: ViTForImageClassification,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    support_type: str,
    rank: int,
    calibration_batches: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    """Calibrate exact compact QK cores and build task-aware supports."""
    if support_type not in {"grad_svd", "skew_grad"}:
        raise ValueError("Expected a task-aware support type.")
    if calibration_batches <= 0:
        raise ValueError("calibration_batches must be positive.")

    for index, block in enumerate(model.encoder.blocks):
        block.attention = QKCentralCalibrationAttention(block.attention)

    for parameter in model.parameters():
        parameter.requires_grad = False
    calibration_layers = [block.attention for block in model.encoder.blocks]
    for layer in calibration_layers:
        layer.central_operators.requires_grad = True

    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    used_batches = 0
    for images, labels in calibration_loader:
        if used_batches >= calibration_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = criterion(model(images), labels) / calibration_batches
        loss.backward()
        used_batches += 1
    if used_batches != calibration_batches:
        raise ValueError(
            f"Requested {calibration_batches} calibration batches, got {used_batches}."
        )

    left_supports: list[torch.Tensor] = []
    right_supports: list[torch.Tensor] = []
    for index, layer in enumerate(calibration_layers):
        gradient = layer.central_operators.grad
        if gradient is None:
            raise RuntimeError(f"No compact-core gradient for layer {index}.")
        left, right = supports_from_gradient(
            gradient.detach(),
            layer.base_singular_values.detach(),
            support_type=support_type,
            rank=rank,
        )
        left_supports.append(left.cpu())
        right_supports.append(right.cpu())
        layer.central_operators.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return left_supports, right_supports, elapsed

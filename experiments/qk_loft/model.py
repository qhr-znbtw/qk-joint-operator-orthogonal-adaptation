"""Build ViT-B/16 with fixed-support joint-QK LOFT adapters."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from experiments.qk_loft.attention import QKLOFTAttention
from experiments.qk_loft.config import QK_LOFT_RANK
from experiments.qk_loft.supports import (
    build_gradient_supports,
    build_principal_supports,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import ViTForImageClassification


class ViTQKLOFTClassifier(ViTForImageClassification):
    """A frozen ViT whose per-head joint QK operators use QK-LOFT."""

    def __init__(self, num_classes: int, *, side: str) -> None:
        super().__init__(num_classes=num_classes)
        if side not in {"left", "right", "bilateral"}:
            raise ValueError(f"Unknown side: {side}")
        self.side = side
        self.adapters_initialized = False

    def initialize_adapters(
        self,
        left_supports: list[torch.Tensor],
        right_supports: list[torch.Tensor],
    ) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-LOFT adapters have already been initialized.")
        expected = self.config.num_layers
        if len(left_supports) != expected or len(right_supports) != expected:
            raise ValueError(f"Expected {expected} supports on each side.")
        for index, block in enumerate(self.encoder.blocks):
            block.attention = QKLOFTAttention(
                block.attention,
                side=self.side,
                left_support=(
                    left_supports[index]
                    if self.side in {"left", "bilateral"}
                    else None
                ),
                right_support=(
                    right_supports[index]
                    if self.side in {"right", "bilateral"}
                    else None
                ),
            )
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKLOFTAttention]:
        layers = [
            module for module in self.modules() if isinstance(module, QKLOFTAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one QK-LOFT adapter per Transformer layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for layer in self.adapter_layers():
            if layer.left_skew_values is not None:
                layer.left_skew_values.requires_grad = True
            if layer.right_skew_values is not None:
                layer.right_skew_values.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _cache_all_transforms(self) -> list[QKLOFTAttention]:
        layers = self.adapter_layers()
        if not layers:
            raise RuntimeError("QK-LOFT adapters must be initialized first.")

        left_skew = None
        right_skew = None
        if self.side in {"left", "bilateral"}:
            left_skew = torch.cat(
                [layer.left_skew_matrices().float() for layer in layers], dim=0
            )
        if self.side in {"right", "bilateral"}:
            right_skew = torch.cat(
                [layer.right_skew_matrices().float() for layer in layers], dim=0
            )
        available = left_skew if left_skew is not None else right_skew
        if available is None:
            raise RuntimeError("No active QK-LOFT transform.")
        all_skew = torch.cat(
            [item for item in (left_skew, right_skew) if item is not None], dim=0
        )
        with torch.autocast(device_type=all_skew.device.type, enabled=False):
            identity = torch.eye(
                layers[0].rank,
                device=all_skew.device,
                dtype=torch.float32,
            )
            all_transforms = torch.linalg.solve(
                identity + all_skew,
                identity - all_skew,
            )

        count = available.shape[0]
        left_transforms = (
            all_transforms[:count] if left_skew is not None else None
        )
        right_transforms = (
            all_transforms[-count:] if right_skew is not None else None
        )
        offset = 0
        for layer in layers:
            next_offset = offset + layer.num_heads
            layer.set_transform_cache(
                None
                if left_transforms is None
                else left_transforms[offset:next_offset],
                None
                if right_transforms is None
                else right_transforms[offset:next_offset],
            )
            offset = next_offset
        return layers

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.adapters_initialized:
            return super().forward(images)
        layers = self._cache_all_transforms()
        try:
            return super().forward(images)
        finally:
            for layer in layers:
                layer.clear_transform_cache()


def build_qk_loft_model(
    num_classes: int,
    support_type: str,
    side: str,
    *,
    device: torch.device,
    criterion: nn.Module,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None,
    calibration_batches: int,
) -> tuple[ViTQKLOFTClassifier, dict[str, object]]:
    """Load pretrained ViT, construct supports, and inject QK-LOFT."""
    model = ViTQKLOFTClassifier(num_classes=num_classes, side=side)
    loaded_tensors = load_pretrained_vit(model)

    if support_type == "principal":
        left, right, support_seconds = build_principal_supports(
            model,
            rank=QK_LOFT_RANK,
        )
        used_calibration_batches = 0
    else:
        if calibration_loader is None:
            raise ValueError(f"{support_type} requires a calibration loader.")
        left, right, support_seconds = build_gradient_supports(
            model,
            calibration_loader,
            criterion,
            support_type=support_type,
            rank=QK_LOFT_RANK,
            calibration_batches=calibration_batches,
            device=device,
        )
        used_calibration_batches = calibration_batches

    model.initialize_adapters(left, right)
    metadata: dict[str, object] = {
        "loaded_tensors": loaded_tensors,
        "support_construction_seconds": support_seconds,
        "calibration_batches": used_calibration_batches,
        "num_heads_adapted": model.config.num_layers * model.config.num_heads,
    }
    return model, metadata

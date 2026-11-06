"""ViT-B/16 with global external and per-head internal bilateral QK-BOFT."""

from __future__ import annotations

import torch

from experiments.qk_boft_internal_external.attention import (
    QKBOFTInternalExternalAttention,
)
from experiments.qk_boft_internal_external.config import (
    EXTERNAL_BOFT_BLOCK_SIZE,
    EXTERNAL_BOFT_NUM_FACTORS,
)
from experiments.qk_boft_internal_external.external_boft import (
    GlobalBilateralExternalBOFT,
)
from experiments.qk_boft_left.config import BOFT_BLOCK_SIZE, BOFT_NUM_FACTORS
from vit_core.vit_backbone import ViTForImageClassification


class ViTQKBOFTInternalExternalClassifier(ViTForImageClassification):
    """Combine shared ambient BOFT with independent compact BOFT rotations."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
        self.adapters_initialized = False
        self.external_boft: GlobalBilateralExternalBOFT | None = None

    def initialize_adapters(self) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-BOFT adapters have already been initialized.")
        self.external_boft = GlobalBilateralExternalBOFT(
            dimension=self.config.hidden_size,
            block_size=EXTERNAL_BOFT_BLOCK_SIZE,
            num_factors=EXTERNAL_BOFT_NUM_FACTORS,
        )
        for block in self.encoder.blocks:
            block.attention = QKBOFTInternalExternalAttention(
                block.attention,
                block_size=BOFT_BLOCK_SIZE,
                num_factors=BOFT_NUM_FACTORS,
                external_block_size=EXTERNAL_BOFT_BLOCK_SIZE,
                external_num_factors=EXTERNAL_BOFT_NUM_FACTORS,
            )
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKBOFTInternalExternalAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKBOFTInternalExternalAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one internal-external adapter per layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        if self.external_boft is None:
            raise RuntimeError("External BOFT must be initialized first.")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for layer in self.adapter_layers():
            layer.skew_parameters.requires_grad = True
            layer.right_skew_parameters.requires_grad = True
        self.external_boft.left_skew_parameters.requires_grad = True
        self.external_boft.right_skew_parameters.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _cache_all_rotations(self) -> list[QKBOFTInternalExternalAttention]:
        layers = self.adapter_layers()
        if not layers or self.external_boft is None:
            raise RuntimeError("Internal and external BOFT must be initialized.")

        with torch.autocast(
            device_type=layers[0].skew_parameters.device.type,
            enabled=False,
        ):
            left_skew = torch.cat(
                [layer.left_skew_matrices().float() for layer in layers], dim=0
            )
            right_skew = torch.cat(
                [layer.right_skew_matrices().float() for layer in layers], dim=0
            )
            all_skew = torch.cat((left_skew, right_skew), dim=0)
            identity = torch.eye(
                layers[0].block_size,
                dtype=all_skew.dtype,
                device=all_skew.device,
            )
            all_rotations = torch.linalg.solve(
                identity + 0.5 * all_skew,
                identity - 0.5 * all_skew,
            )
            external_left = self.external_boft.left_orthogonal_blocks()
            external_right = self.external_boft.right_orthogonal_blocks()

        split = left_skew.shape[0]
        left_rotations = all_rotations[:split]
        right_rotations = all_rotations[split:]
        offset = 0
        for layer in layers:
            next_offset = offset + layer.num_heads
            layer.set_rotation_cache(
                left_rotations[offset:next_offset],
                right_rotations[offset:next_offset],
            )
            offset = next_offset

        internal_factors = [layer.internal_adapted_factors() for layer in layers]
        internal_left = torch.stack([pair[0] for pair in internal_factors])
        internal_right = torch.stack([pair[1] for pair in internal_factors])
        adapted_left = self.external_boft.rotate_feature_rows(
            internal_left,
            external_left,
        )
        adapted_right = self.external_boft.rotate_feature_rows(
            internal_right,
            external_right,
        )
        for layer_index, layer in enumerate(layers):
            layer.set_adapted_factor_cache(
                adapted_left[layer_index],
                adapted_right[layer_index],
            )
        return layers

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.adapters_initialized:
            return super().forward(images)
        layers = self._cache_all_rotations()
        try:
            return super().forward(images)
        finally:
            for layer in layers:
                layer.clear_rotation_cache()
                layer.clear_external_rotation_cache()
                layer.clear_adapted_factor_cache()

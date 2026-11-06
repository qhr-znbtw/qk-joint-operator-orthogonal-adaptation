"""ViT-B/16 with full-spectrum bilateral joint-QK BOFT."""

import torch

from experiments.qk_boft_bilateral.attention import QKBOFTBilateralAttention
from experiments.qk_boft_left.config import BOFT_BLOCK_SIZE, BOFT_NUM_FACTORS
from vit_core.vit_backbone import ViTForImageClassification


class ViTQKBOFTBilateralClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train independent left/right BOFT products."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
        self.adapters_initialized = False

    def initialize_adapters(self) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-BOFT adapters have already been initialized.")
        for block in self.encoder.blocks:
            block.attention = QKBOFTBilateralAttention(
                block.attention,
                block_size=BOFT_BLOCK_SIZE,
                num_factors=BOFT_NUM_FACTORS,
            )
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKBOFTBilateralAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKBOFTBilateralAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one bilateral QK-BOFT adapter per layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for layer in self.adapter_layers():
            layer.skew_parameters.requires_grad = True
            layer.right_skew_parameters.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _cache_all_rotations(self) -> list[QKBOFTBilateralAttention]:
        layers = self.adapter_layers()
        if not layers:
            raise RuntimeError("QK-BOFT adapters must be initialized first.")
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

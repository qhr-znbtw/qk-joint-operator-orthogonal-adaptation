"""ViT-B/16 with full-spectrum bilateral joint-QK PSOFT."""

import torch

from experiments.qk_psoft_bilateral.attention import QKPSOFTBilateralAttention
from vit_core.vit_backbone import ViTForImageClassification


class ViTQKPSOFTBilateralClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train two dense rotations per QK head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
        self.adapters_initialized = False

    def initialize_adapters(self) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-PSOFT adapters have already been initialized.")
        for block in self.encoder.blocks:
            block.attention = QKPSOFTBilateralAttention(block.attention)
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKPSOFTBilateralAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKPSOFTBilateralAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one QK-PSOFT adapter per Transformer layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for layer in self.adapter_layers():
            layer.left_skew_parameters.requires_grad = True
            layer.right_skew_parameters.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _cache_all_rotations(self) -> list[QKPSOFTBilateralAttention]:
        layers = self.adapter_layers()
        if not layers:
            raise RuntimeError("QK-PSOFT adapters must be initialized first.")

        with torch.autocast(
            device_type=layers[0].left_skew_parameters.device.type,
            enabled=False,
        ):
            left_skew = torch.cat(
                [layer.left_skew_matrices().float() for layer in layers],
                dim=0,
            )
            right_skew = torch.cat(
                [layer.right_skew_matrices().float() for layer in layers],
                dim=0,
            )
            all_skew = torch.cat((left_skew, right_skew), dim=0)
            identity = torch.eye(
                layers[0].head_size,
                dtype=all_skew.dtype,
                device=all_skew.device,
            )
            all_rotations = torch.linalg.solve(
                identity + 0.5 * all_skew,
                identity - 0.5 * all_skew,
            )
        num_rotations_per_side = left_skew.shape[0]
        left_rotations = all_rotations[:num_rotations_per_side]
        right_rotations = all_rotations[num_rotations_per_side:]

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


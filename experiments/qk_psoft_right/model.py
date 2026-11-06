"""ViT-B/16 with full-spectrum right-sided joint-QK PSOFT."""

import torch

from experiments.qk_psoft_right.attention import QKPSOFTRightAttention
from vit_core.vit_backbone import ViTForImageClassification


class ViTQKPSOFTRightClassifier(ViTForImageClassification):
    """Freeze ViT-B/16 and train one dense right rotation per QK head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
        self.adapters_initialized = False

    def initialize_adapters(self) -> None:
        """Decompose pretrained QK operators and install trainable rotations."""
        if self.adapters_initialized:
            raise RuntimeError("QK-PSOFT adapters have already been initialized.")
        for block in self.encoder.blocks:
            block.attention = QKPSOFTRightAttention(block.attention)
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKPSOFTRightAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKPSOFTRightAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one QK-PSOFT adapter per Transformer layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        """Train only Cayley coordinates and the task classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for layer in self.adapter_layers():
            layer.skew_parameters.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def _cache_all_rotations(self) -> list[QKPSOFTRightAttention]:
        layers = self.adapter_layers()
        if not layers:
            raise RuntimeError("QK-PSOFT adapters must be initialized first.")

        with torch.autocast(
            device_type=layers[0].skew_parameters.device.type,
            enabled=False,
        ):
            skew = torch.cat(
                [layer.skew_matrices().float() for layer in layers],
                dim=0,
            )
            identity = torch.eye(
                layers[0].head_size,
                dtype=skew.dtype,
                device=skew.device,
            )
            rotations = torch.linalg.solve(
                identity + 0.5 * skew,
                identity - 0.5 * skew,
            )

        offset = 0
        for layer in layers:
            next_offset = offset + layer.num_heads
            layer.set_rotation_cache(rotations[offset:next_offset])
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


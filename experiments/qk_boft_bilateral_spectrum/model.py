"""ViT-B/16 with bilateral BOFT and unconstrained positive spectrum."""

from experiments.qk_boft_bilateral.model import ViTQKBOFTBilateralClassifier
from experiments.qk_boft_bilateral_spectrum.attention import (
    QKBOFTBilateralSpectrumAttention,
)
from experiments.qk_boft_left.config import BOFT_BLOCK_SIZE, BOFT_NUM_FACTORS


class ViTQKBOFTBilateralSpectrumClassifier(ViTQKBOFTBilateralClassifier):
    """Train two BOFT rotations and unconstrained log-scales per head."""

    def initialize_adapters(self) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-BOFT adapters have already been initialized.")
        for block in self.encoder.blocks:
            block.attention = QKBOFTBilateralSpectrumAttention(
                block.attention,
                block_size=BOFT_BLOCK_SIZE,
                num_factors=BOFT_NUM_FACTORS,
            )
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKBOFTBilateralSpectrumAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKBOFTBilateralSpectrumAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one spectral QK-BOFT adapter per layer.")
        return layers

    def set_trainable_parameters(self) -> None:
        super().set_trainable_parameters()
        for layer in self.adapter_layers():
            layer.spectrum_log_scales.requires_grad = True

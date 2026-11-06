"""ViT-B/16 with full-spectrum right-sided joint-QK BOFT."""

from experiments.qk_boft_left.model import ViTQKBOFTLeftClassifier
from experiments.qk_boft_right.attention import QKBOFTRightAttention
from experiments.qk_boft_left.config import BOFT_BLOCK_SIZE, BOFT_NUM_FACTORS


class ViTQKBOFTRightClassifier(ViTQKBOFTLeftClassifier):
    """Freeze ViT-B/16 and train one right BOFT product per QK head."""

    def initialize_adapters(self) -> None:
        if self.adapters_initialized:
            raise RuntimeError("QK-BOFT adapters have already been initialized.")
        for block in self.encoder.blocks:
            block.attention = QKBOFTRightAttention(
                block.attention,
                block_size=BOFT_BLOCK_SIZE,
                num_factors=BOFT_NUM_FACTORS,
            )
        self.adapters_initialized = True
        self.set_trainable_parameters()

    def adapter_layers(self) -> list[QKBOFTRightAttention]:
        layers = [
            module
            for module in self.modules()
            if isinstance(module, QKBOFTRightAttention)
        ]
        if self.adapters_initialized and len(layers) != self.config.num_layers:
            raise RuntimeError("Expected one right QK-BOFT adapter per layer.")
        return layers

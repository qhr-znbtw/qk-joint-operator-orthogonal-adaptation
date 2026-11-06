"""Bilateral BOFT with unconstrained positive spectrum learning."""

import torch
from torch import nn

from experiments.qk_boft_bilateral.attention import QKBOFTBilateralAttention


class QKBOFTBilateralSpectrumAttention(QKBOFTBilateralAttention):
    """Learn spectral shape together with independent left/right rotations.

    The learned positive spectrum is ``sigma' = sigma0 * exp(alpha)``. Unlike
    the earlier norm-preserving variant, this permits both relative spectral
    reshaping and a change in the overall attention-logit scale.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spectrum_log_scales = nn.Parameter(
            torch.zeros_like(self.sqrt_singular_values)
        )

    def base_singular_values(self) -> torch.Tensor:
        return self.sqrt_singular_values.float().square()

    def learned_singular_values(self) -> torch.Tensor:
        """Return positive spectra without centering or norm restoration."""
        with torch.autocast(
            device_type=self.spectrum_log_scales.device.type,
            enabled=False,
        ):
            base = self.base_singular_values()
            return base * self.spectrum_log_scales.float().exp()

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        singular_values = self.learned_singular_values()
        sqrt_singular_values = singular_values.clamp_min(0).sqrt()
        left = self.rotate_features_with_blocks(
            self.left_vectors, self.left_orthogonal_blocks()
        )
        right = self.rotate_features_with_blocks(
            self.right_vectors, self.right_orthogonal_blocks()
        )
        left = left * sqrt_singular_values[:, None, :]
        right = right * sqrt_singular_values[:, None, :]
        return left, right

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, rank={self.head_size}, "
            f"block={self.block_size}, factors={self.num_factors}, "
            "side=bilateral, spectrum=learnable_unconstrained_log_scale"
        )

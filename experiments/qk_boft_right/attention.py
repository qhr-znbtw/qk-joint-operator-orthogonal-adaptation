"""Right-sided BOFT over the complete compact spectrum of one QK layer."""

import torch

from experiments.qk_boft_left.attention import QKBOFTLeftAttention


class QKBOFTRightAttention(QKBOFTLeftAttention):
    """Apply ``M' = U diag(s) R_boft.T V.T`` to each attention head."""

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.left_vectors * self.sqrt_singular_values[:, None, :]
        right = self.rotate_features(self.right_vectors)
        right = right * self.sqrt_singular_values[:, None, :]
        return left, right

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, rank={self.head_size}, "
            f"block={self.block_size}, factors={self.num_factors}, side=right"
        )

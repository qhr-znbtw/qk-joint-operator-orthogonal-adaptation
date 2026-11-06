"""Bilateral BOFT over the complete compact spectrum of one QK layer."""

import torch
from torch import nn

from experiments.qk_boft_left.attention import QKBOFTLeftAttention


class QKBOFTBilateralAttention(QKBOFTLeftAttention):
    """Apply ``M' = U R_left diag(s) R_right.T V.T`` per head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.right_skew_parameters = nn.Parameter(
            torch.zeros_like(self.skew_parameters)
        )
        self._right_rotation_cache: torch.Tensor | None = None

    @property
    def left_skew_parameters(self) -> nn.Parameter:
        return self.skew_parameters

    def left_skew_matrices(self) -> torch.Tensor:
        return self.skew_matrices()

    def right_skew_matrices(self) -> torch.Tensor:
        return self.skew_matrices_from(self.right_skew_parameters)

    def left_orthogonal_blocks(self) -> torch.Tensor:
        return self.orthogonal_blocks()

    def right_orthogonal_blocks(self) -> torch.Tensor:
        if self._right_rotation_cache is not None:
            return self._right_rotation_cache
        return self.cayley_blocks(self.right_skew_matrices())

    def set_rotation_cache(
        self,
        left_rotations: torch.Tensor,
        right_rotations: torch.Tensor,
    ) -> None:
        expected = (
            self.num_heads,
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if left_rotations.shape != expected or right_rotations.shape != expected:
            raise ValueError(
                f"Both rotation caches must have shape {expected}; got "
                f"{tuple(left_rotations.shape)} and {tuple(right_rotations.shape)}."
            )
        self._rotation_cache = left_rotations
        self._right_rotation_cache = right_rotations

    def clear_rotation_cache(self) -> None:
        self._rotation_cache = None
        self._right_rotation_cache = None

    def dense_left_orthogonal_matrices(self) -> torch.Tensor:
        identity = torch.eye(
            self.head_size,
            dtype=self.skew_parameters.dtype,
            device=self.skew_parameters.device,
        ).expand(self.num_heads, -1, -1)
        return self.rotate_features_with_blocks(
            identity, self.left_orthogonal_blocks()
        )

    def dense_right_orthogonal_matrices(self) -> torch.Tensor:
        identity = torch.eye(
            self.head_size,
            dtype=self.right_skew_parameters.dtype,
            device=self.right_skew_parameters.device,
        ).expand(self.num_heads, -1, -1)
        return self.rotate_features_with_blocks(
            identity, self.right_orthogonal_blocks()
        )

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.rotate_features_with_blocks(
            self.left_vectors, self.left_orthogonal_blocks()
        )
        left = left * self.sqrt_singular_values[:, None, :]
        right = self.rotate_features_with_blocks(
            self.right_vectors, self.right_orthogonal_blocks()
        )
        right = right * self.sqrt_singular_values[:, None, :]
        return left, right

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, rank={self.head_size}, "
            f"block={self.block_size}, factors={self.num_factors}, side=bilateral"
        )

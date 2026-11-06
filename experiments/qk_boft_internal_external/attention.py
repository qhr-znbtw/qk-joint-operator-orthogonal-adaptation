"""Per-head internal bilateral BOFT plus cached global external BOFT."""

from __future__ import annotations

import torch

from experiments.qk_boft_bilateral.attention import QKBOFTBilateralAttention
from experiments.qk_boft_left.attention import build_coordinate_permutation


class QKBOFTInternalExternalAttention(QKBOFTBilateralAttention):
    """Apply global ambient rotations outside per-head compact rotations."""

    def __init__(
        self,
        *args,
        external_block_size: int = 24,
        external_num_factors: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.hidden_size % external_block_size != 0:
            raise ValueError("hidden_size must be divisible by external_block_size.")
        self.external_block_size = external_block_size
        self.external_num_factors = external_num_factors
        self.external_num_blocks = self.hidden_size // external_block_size

        permutations = []
        inverse_permutations = []
        for factor_index in range(external_num_factors):
            order = build_coordinate_permutation(
                self.hidden_size,
                external_block_size,
                factor_index,
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(self.hidden_size)
            permutations.append(order)
            inverse_permutations.append(inverse)
        self.register_buffer(
            "external_permutations",
            torch.stack(permutations),
            persistent=False,
        )
        self.register_buffer(
            "external_inverse_permutations",
            torch.stack(inverse_permutations),
            persistent=False,
        )
        self._external_left_cache: torch.Tensor | None = None
        self._external_right_cache: torch.Tensor | None = None
        self._adapted_factor_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_external_rotation_cache(
        self,
        left_rotations: torch.Tensor,
        right_rotations: torch.Tensor,
    ) -> None:
        expected = (
            self.external_num_factors,
            self.external_num_blocks,
            self.external_block_size,
            self.external_block_size,
        )
        if left_rotations.shape != expected or right_rotations.shape != expected:
            raise ValueError(
                f"External rotation caches must both have shape {expected}."
            )
        self._external_left_cache = left_rotations
        self._external_right_cache = right_rotations

    def clear_external_rotation_cache(self) -> None:
        self._external_left_cache = None
        self._external_right_cache = None

    def set_adapted_factor_cache(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> None:
        expected = (
            self.num_heads,
            self.hidden_size + 1,
            self.head_size,
        )
        if left.shape != expected or right.shape != expected:
            raise ValueError(f"Adapted factors must both have shape {expected}.")
        self._adapted_factor_cache = (left, right)

    def clear_adapted_factor_cache(self) -> None:
        self._adapted_factor_cache = None

    def _apply_external_rotation(
        self,
        factors: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Left-multiply feature rows while leaving the bias row unchanged."""
        feature_rows = factors[:, : self.hidden_size, :].transpose(-1, -2)
        outputs = feature_rows
        rotations = rotations.to(dtype=factors.dtype)
        for factor_index in range(self.external_num_factors):
            order = self.external_permutations[factor_index]
            inverse = self.external_inverse_permutations[factor_index]
            permuted = outputs.index_select(-1, order)
            blocks = permuted.reshape(
                self.num_heads,
                self.head_size,
                self.external_num_blocks,
                self.external_block_size,
            )
            rotated = torch.einsum(
                "hrki,kij->hrkj",
                blocks,
                rotations[factor_index],
            )
            outputs = rotated.flatten(-2).index_select(-1, inverse)
        rotated_features = outputs.transpose(-1, -2)
        bias_row = factors[:, self.hidden_size :, :]
        return torch.cat((rotated_features, bias_row), dim=1)

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._adapted_factor_cache is not None:
            return self._adapted_factor_cache
        if self._external_left_cache is None or self._external_right_cache is None:
            raise RuntimeError("Global external BOFT rotations are not cached.")
        left, right = self.internal_adapted_factors()
        left = self._apply_external_rotation(left, self._external_left_cache)
        right = self._apply_external_rotation(right, self._external_right_cache)
        return left, right

    def internal_adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-head bilateral factors before ambient rotation."""
        return super().adapted_factors()

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, rank={self.head_size}, "
            f"internal_block={self.block_size}, internal_factors={self.num_factors}, "
            f"external_block={self.external_block_size}, "
            f"external_factors={self.external_num_factors}"
        )

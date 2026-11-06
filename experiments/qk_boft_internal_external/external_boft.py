"""Two globally shared ambient-space BOFT rotations."""

from __future__ import annotations

import torch
from torch import nn

from experiments.qk_boft_left.attention import build_coordinate_permutation


class GlobalBilateralExternalBOFT(nn.Module):
    """Learn independent global left/right BOFT products in 768 dimensions."""

    def __init__(
        self,
        dimension: int = 768,
        block_size: int = 24,
        num_factors: int = 6,
    ) -> None:
        super().__init__()
        if dimension % block_size != 0:
            raise ValueError("dimension must be divisible by block_size.")
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer.")
        if num_factors <= 0:
            raise ValueError("num_factors must be positive.")

        self.dimension = dimension
        self.block_size = block_size
        self.num_factors = num_factors
        self.num_blocks = dimension // block_size
        independent_entries = block_size * (block_size - 1) // 2
        parameter_shape = (
            num_factors,
            self.num_blocks,
            independent_entries,
        )
        self.left_skew_parameters = nn.Parameter(torch.zeros(parameter_shape))
        self.right_skew_parameters = nn.Parameter(torch.zeros(parameter_shape))

        upper_row, upper_col = torch.triu_indices(
            block_size,
            block_size,
            offset=1,
        )
        self.register_buffer("upper_row", upper_row, persistent=False)
        self.register_buffer("upper_col", upper_col, persistent=False)

        permutations = []
        inverse_permutations = []
        for factor_index in range(num_factors):
            order = build_coordinate_permutation(
                dimension,
                block_size,
                factor_index,
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(dimension)
            permutations.append(order)
            inverse_permutations.append(inverse)
        self.register_buffer(
            "permutations",
            torch.stack(permutations),
            persistent=False,
        )
        self.register_buffer(
            "inverse_permutations",
            torch.stack(inverse_permutations),
            persistent=False,
        )

    def skew_matrices_from(self, parameters: torch.Tensor) -> torch.Tensor:
        expected = self.left_skew_parameters.shape
        if parameters.shape != expected:
            raise ValueError(
                f"Expected external BOFT parameters with shape {tuple(expected)}, "
                f"got {tuple(parameters.shape)}."
            )
        shape = (
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        skew = parameters.new_zeros(shape)
        skew[..., self.upper_row, self.upper_col] = parameters
        skew[..., self.upper_col, self.upper_row] = -parameters
        return skew

    def cayley_blocks(self, skew: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=skew.device.type, enabled=False):
            skew = skew.float()
            identity = torch.eye(
                self.block_size,
                dtype=skew.dtype,
                device=skew.device,
            )
            return torch.linalg.solve(
                identity + 0.5 * skew,
                identity - 0.5 * skew,
            )

    def left_orthogonal_blocks(self) -> torch.Tensor:
        return self.cayley_blocks(
            self.skew_matrices_from(self.left_skew_parameters)
        )

    def right_orthogonal_blocks(self) -> torch.Tensor:
        return self.cayley_blocks(
            self.skew_matrices_from(self.right_skew_parameters)
        )

    def rotate_last_dimension(
        self,
        inputs: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a shared BOFT product to row vectors along the last axis."""
        if inputs.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected final dimension {self.dimension}, got {inputs.shape[-1]}."
            )
        expected = (
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if rotations.shape != expected:
            raise ValueError(
                f"Expected rotation blocks with shape {expected}, "
                f"got {tuple(rotations.shape)}."
            )

        outputs = inputs
        rotations = rotations.to(dtype=inputs.dtype)
        for factor_index in range(self.num_factors):
            order = self.permutations[factor_index]
            inverse = self.inverse_permutations[factor_index]
            permuted = outputs.index_select(-1, order)
            blocks = permuted.reshape(
                *permuted.shape[:-1],
                self.num_blocks,
                self.block_size,
            )
            rotated = torch.einsum(
                "...ki,kij->...kj",
                blocks,
                rotations[factor_index],
            )
            outputs = rotated.flatten(-2).index_select(-1, inverse)
        return outputs

    def rotate_feature_rows(
        self,
        factors: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Left-rotate 768 feature rows and preserve one final bias row.

        ``factors`` may have any leading dimensions and must end in
        ``[dimension + 1, rank]``.  Stacking every layer and head before this
        call lets the globally shared rotation run in one vectorized pass.
        """
        if factors.shape[-2] != self.dimension + 1:
            raise ValueError(
                f"Expected {self.dimension + 1} augmented rows, "
                f"got {factors.shape[-2]}."
            )
        feature_rows = factors[..., : self.dimension, :].transpose(-1, -2)
        rotated_features = self.rotate_last_dimension(
            feature_rows,
            rotations,
        ).transpose(-1, -2)
        return torch.cat(
            (rotated_features, factors[..., self.dimension :, :]),
            dim=-2,
        )

    def dense_left_row_rotation(self) -> torch.Tensor:
        identity = torch.eye(
            self.dimension,
            dtype=self.left_skew_parameters.dtype,
            device=self.left_skew_parameters.device,
        )
        return self.rotate_last_dimension(identity, self.left_orthogonal_blocks())

    def dense_right_row_rotation(self) -> torch.Tensor:
        identity = torch.eye(
            self.dimension,
            dtype=self.right_skew_parameters.dtype,
            device=self.right_skew_parameters.device,
        )
        return self.rotate_last_dimension(identity, self.right_orthogonal_blocks())

    def extra_repr(self) -> str:
        return (
            f"dimension={self.dimension}, block={self.block_size}, "
            f"factors={self.num_factors}, sharing=global"
        )

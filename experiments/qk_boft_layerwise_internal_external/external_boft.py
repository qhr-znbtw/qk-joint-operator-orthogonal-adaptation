"""Independent ambient BOFT pairs for each Transformer layer."""

from __future__ import annotations

import torch
from torch import nn

from experiments.qk_boft_left.attention import build_coordinate_permutation


class LayerwiseBilateralExternalBOFT(nn.Module):
    """Learn one left/right BOFT pair per layer, shared across its heads."""

    def __init__(
        self,
        num_layers: int = 12,
        dimension: int = 768,
        block_size: int = 24,
        num_factors: int = 6,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if dimension % block_size != 0:
            raise ValueError("dimension must be divisible by block_size.")
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer.")
        if num_factors <= 0:
            raise ValueError("num_factors must be positive.")

        self.num_layers = num_layers
        self.dimension = dimension
        self.block_size = block_size
        self.num_factors = num_factors
        self.num_blocks = dimension // block_size
        independent_entries = block_size * (block_size - 1) // 2
        parameter_shape = (
            num_layers,
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
                f"Expected layer-wise BOFT parameters with shape {tuple(expected)}, "
                f"got {tuple(parameters.shape)}."
            )
        shape = (
            self.num_layers,
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

    def rotate_feature_rows(
        self,
        factors: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the corresponding layer rotation and preserve each bias row.

        ``factors`` must have shape ``[layers, heads, dimension + 1, rank]``.
        """
        if factors.ndim != 4:
            raise ValueError("factors must have [layers, heads, rows, rank].")
        if factors.shape[0] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {factors.shape[0]}."
            )
        if factors.shape[-2] != self.dimension + 1:
            raise ValueError(
                f"Expected {self.dimension + 1} augmented rows, "
                f"got {factors.shape[-2]}."
            )
        expected = (
            self.num_layers,
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if rotations.shape != expected:
            raise ValueError(
                f"Expected rotations with shape {expected}, "
                f"got {tuple(rotations.shape)}."
            )

        outputs = factors[..., : self.dimension, :].transpose(-1, -2)
        rotations = rotations.to(dtype=factors.dtype)
        for factor_index in range(self.num_factors):
            order = self.permutations[factor_index]
            inverse = self.inverse_permutations[factor_index]
            permuted = outputs.index_select(-1, order)
            blocks = permuted.reshape(
                self.num_layers,
                factors.shape[1],
                factors.shape[-1],
                self.num_blocks,
                self.block_size,
            )
            rotated = torch.einsum(
                "lhrki,lkij->lhrkj",
                blocks,
                rotations[:, factor_index],
            )
            outputs = rotated.flatten(-2).index_select(-1, inverse)

        rotated_features = outputs.transpose(-1, -2)
        return torch.cat(
            (rotated_features, factors[..., self.dimension :, :]),
            dim=-2,
        )

    def extra_repr(self) -> str:
        return (
            f"layers={self.num_layers}, dimension={self.dimension}, "
            f"block={self.block_size}, factors={self.num_factors}, "
            "sharing=within_layer_heads"
        )

"""Left-sided BOFT over the complete compact spectrum of one QK layer."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.qk_svd_oft.qk_svd import decompose_attention_qk
from vit_core.vit_backbone import MultiHeadSelfAttention


def build_coordinate_permutation(
    dimension: int,
    block_size: int,
    factor_index: int,
) -> torch.Tensor:
    """Return the fixed binary-butterfly coordinate order for one factor."""
    if block_size <= 0 or block_size % 2 != 0:
        raise ValueError("block_size must be a positive even integer.")
    if dimension % block_size != 0:
        raise ValueError("dimension must be divisible by block_size.")
    micro_block_size = block_size // 2
    num_micro_blocks = dimension // micro_block_size
    stride = 2**factor_index
    group_size = 2 * stride
    if num_micro_blocks % group_size != 0:
        raise ValueError(
            f"Factor {factor_index + 1} requires {num_micro_blocks} "
            f"micro-blocks to be divisible by {group_size}."
        )

    micro_block_order: list[int] = []
    for group_start in range(0, num_micro_blocks, group_size):
        for offset in range(stride):
            micro_block_order.extend(
                (group_start + offset, group_start + offset + stride)
            )

    coordinate_order: list[int] = []
    for micro_block in micro_block_order:
        coordinate_start = micro_block * micro_block_size
        coordinate_order.extend(
            range(coordinate_start, coordinate_start + micro_block_size)
        )
    return torch.tensor(coordinate_order, dtype=torch.long)


class QKBOFTLeftAttention(nn.Module):
    """Apply a trainable butterfly rotation to the left QK singular space.

    The adapted bias-aware operator is

    ``M' = U R_boft diag(s) V.T``

    where ``R_boft`` is the product of fixed-permutation block-orthogonal
    factors.  All 64 compact singular directions remain present.
    """

    def __init__(
        self,
        attention: MultiHeadSelfAttention,
        *,
        block_size: int = 4,
        num_factors: int = 5,
    ) -> None:
        super().__init__()
        if not isinstance(attention, MultiHeadSelfAttention):
            raise TypeError("attention must be a MultiHeadSelfAttention module.")
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer.")
        if num_factors <= 0:
            raise ValueError("num_factors must be positive.")

        self.num_heads = attention.num_heads
        self.head_size = attention.head_size
        self.hidden_size = attention.query.in_features
        self.block_size = block_size
        self.num_factors = num_factors
        if self.head_size % block_size != 0:
            raise ValueError("head_size must be divisible by block_size.")
        self.num_blocks = self.head_size // block_size
        if attention.query.out_features != self.num_heads * self.head_size:
            raise ValueError("Query output width does not match the head layout.")
        if attention.key.in_features != self.hidden_size:
            raise ValueError("Query and key input widths must match.")

        self.query = attention.query
        self.key = attention.key
        self.value = attention.value
        self.output = attention.output

        with torch.no_grad():
            decomposition = decompose_attention_qk(attention)
            if decomposition.singular_values.shape[-1] != self.head_size:
                raise RuntimeError("The compact QK rank must equal the head size.")
            left_vectors = decomposition.left_vectors.detach().clone()
            singular_values = decomposition.singular_values.detach().clone()
            right_vectors = (
                decomposition.right_vectors_transpose.transpose(-1, -2)
                .detach()
                .clone()
            )

        self.register_buffer("left_vectors", left_vectors, persistent=False)
        self.register_buffer(
            "sqrt_singular_values",
            singular_values.clamp_min(0).sqrt(),
            persistent=False,
        )
        self.register_buffer("right_vectors", right_vectors, persistent=False)

        independent_entries = block_size * (block_size - 1) // 2
        self.skew_parameters = nn.Parameter(
            torch.zeros(
                self.num_heads,
                num_factors,
                self.num_blocks,
                independent_entries,
            )
        )
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
                self.head_size,
                block_size,
                factor_index,
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(self.head_size)
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
        self._rotation_cache: torch.Tensor | None = None

    def skew_matrices_from(self, parameters: torch.Tensor) -> torch.Tensor:
        shape = (
            self.num_heads,
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if parameters.shape != self.skew_parameters.shape:
            raise ValueError(
                "BOFT parameters must have shape "
                f"{tuple(self.skew_parameters.shape)}, got {tuple(parameters.shape)}."
            )
        skew = parameters.new_zeros(shape)
        skew[..., self.upper_row, self.upper_col] = parameters
        skew[..., self.upper_col, self.upper_row] = -parameters
        return skew

    def skew_matrices(self) -> torch.Tensor:
        return self.skew_matrices_from(self.skew_parameters)

    def cayley_blocks(self, skew: torch.Tensor) -> torch.Tensor:
        """Map a batch of skew blocks through the scaled Cayley transform."""
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

    def orthogonal_blocks(self) -> torch.Tensor:
        """Map all skew blocks through the scaled Cayley transform."""
        if self._rotation_cache is not None:
            return self._rotation_cache
        return self.cayley_blocks(self.skew_matrices())

    def set_rotation_cache(self, rotations: torch.Tensor) -> None:
        expected = (
            self.num_heads,
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if rotations.shape != expected:
            raise ValueError(
                f"Rotation cache must have shape {expected}, "
                f"got {tuple(rotations.shape)}."
            )
        self._rotation_cache = rotations

    def clear_rotation_cache(self) -> None:
        self._rotation_cache = None

    def rotate_features_with_blocks(
        self,
        inputs: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply supplied butterfly blocks to ``[H, ..., 64]`` data."""
        if inputs.shape[0] != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {inputs.shape[0]}.")
        if inputs.shape[-1] != self.head_size:
            raise ValueError(
                f"Expected final dimension {self.head_size}, got {inputs.shape[-1]}."
            )

        expected = (
            self.num_heads,
            self.num_factors,
            self.num_blocks,
            self.block_size,
            self.block_size,
        )
        if rotations.shape != expected:
            raise ValueError(
                f"Rotation blocks must have shape {expected}, "
                f"got {tuple(rotations.shape)}."
            )
        rotations = rotations.to(dtype=inputs.dtype)
        outputs = inputs
        for factor_index in range(self.num_factors):
            order = self.permutations[factor_index]
            inverse = self.inverse_permutations[factor_index]
            permuted = outputs.index_select(-1, order)
            blocks = permuted.reshape(
                self.num_heads,
                -1,
                self.num_blocks,
                self.block_size,
            )
            rotated_blocks = torch.einsum(
                "hnki,hkij->hnkj",
                blocks,
                rotations[:, factor_index],
            )
            outputs = (
                rotated_blocks.flatten(-2)
                .index_select(-1, inverse)
                .reshape_as(outputs)
            )
        return outputs

    def rotate_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the trainable per-head butterfly product."""
        return self.rotate_features_with_blocks(inputs, self.orthogonal_blocks())

    def dense_orthogonal_matrices(self) -> torch.Tensor:
        identity = torch.eye(
            self.head_size,
            dtype=self.skew_parameters.dtype,
            device=self.skew_parameters.device,
        ).expand(self.num_heads, -1, -1)
        return self.rotate_features(identity)

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return augmented factors whose product is ``U R_boft S V.T``."""
        left = self.rotate_features(self.left_vectors)
        left = left * self.sqrt_singular_values[:, None, :]
        right = self.right_vectors * self.sqrt_singular_values[:, None, :]
        return left, right

    def adapted_query_key_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = self.adapted_factors()
        query_weight = (
            left[:, : self.hidden_size, :]
            .transpose(-1, -2)
            .reshape(self.num_heads * self.head_size, self.hidden_size)
        )
        query_bias = left[:, self.hidden_size, :].reshape(-1)
        key_weight = (
            right[:, : self.hidden_size, :]
            .transpose(-1, -2)
            .reshape(self.num_heads * self.head_size, self.hidden_size)
        )
        key_bias = right[:, self.hidden_size, :].reshape(-1)
        return query_weight, query_bias, key_weight, key_bias

    def attention_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        ones = tokens.new_ones(*tokens.shape[:-1], 1)
        augmented_tokens = torch.cat((tokens, ones), dim=-1)
        left, right = self.adapted_factors()
        left = left.to(dtype=tokens.dtype)
        right = right.to(dtype=tokens.dtype)
        query = torch.einsum("bnd,hdr->bhnr", augmented_tokens, left)
        key = torch.einsum("bnd,hdr->bhnr", augmented_tokens, right)
        return (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_size,
        )
        return tensor.transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_logits(tokens).softmax(dim=-1)
        value = self._split_heads(self.value(tokens))
        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().flatten(2)
        return self.output(context)

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, rank={self.head_size}, "
            f"block_size={self.block_size}, factors={self.num_factors}, side=left"
        )


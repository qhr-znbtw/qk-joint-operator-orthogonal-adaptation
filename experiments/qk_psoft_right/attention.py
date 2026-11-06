"""Right-sided PSOFT for the complete compact spectrum of one QK layer."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.qk_svd_oft.qk_svd import decompose_attention_qk
from vit_core.vit_backbone import MultiHeadSelfAttention


class QKPSOFTRightAttention(nn.Module):
    """Rotate the right singular vectors of every per-head QK operator.

    For each head, the bias-aware operator is adapted as

    ``M' = U diag(s) R.T V.T``

    with fixed ``U``, ``s``, and ``V`` and a trainable dense ``R in SO(d_h)``.
    The forward path uses compact factors whose product is exactly this joint
    operator, while the original modules remain frozen for provenance.
    """

    def __init__(self, attention: MultiHeadSelfAttention) -> None:
        super().__init__()
        if not isinstance(attention, MultiHeadSelfAttention):
            raise TypeError("attention must be a MultiHeadSelfAttention module.")

        self.num_heads = attention.num_heads
        self.head_size = attention.head_size
        self.hidden_size = attention.query.in_features
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

        independent_entries = self.head_size * (self.head_size - 1) // 2
        self.skew_parameters = nn.Parameter(
            torch.zeros(self.num_heads, independent_entries)
        )
        upper_row, upper_col = torch.triu_indices(
            self.head_size,
            self.head_size,
            offset=1,
        )
        self.register_buffer("upper_row", upper_row, persistent=False)
        self.register_buffer("upper_col", upper_col, persistent=False)
        self._rotation_cache: torch.Tensor | None = None

    def skew_matrices(self) -> torch.Tensor:
        """Construct one dense skew-symmetric generator per head."""
        skew = self.skew_parameters.new_zeros(
            self.num_heads,
            self.head_size,
            self.head_size,
        )
        skew[:, self.upper_row, self.upper_col] = self.skew_parameters
        skew[:, self.upper_col, self.upper_row] = -self.skew_parameters
        return skew

    def orthogonal_matrices(self) -> torch.Tensor:
        """Return scaled-Cayley rotations, initialized exactly at identity."""
        if self._rotation_cache is not None:
            return self._rotation_cache

        with torch.autocast(
            device_type=self.skew_parameters.device.type,
            enabled=False,
        ):
            skew = self.skew_matrices().float()
            identity = torch.eye(
                self.head_size,
                dtype=skew.dtype,
                device=skew.device,
            )
            return torch.linalg.solve(
                identity + 0.5 * skew,
                identity - 0.5 * skew,
            )

    def set_rotation_cache(self, rotations: torch.Tensor) -> None:
        expected = (self.num_heads, self.head_size, self.head_size)
        if rotations.shape != expected:
            raise ValueError(
                f"Rotation cache must have shape {expected}, "
                f"got {tuple(rotations.shape)}."
            )
        self._rotation_cache = rotations

    def clear_rotation_cache(self) -> None:
        self._rotation_cache = None

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q/K factors whose product is ``U S R.T V.T``."""
        rotations = self.orthogonal_matrices()
        left = self.left_vectors * self.sqrt_singular_values[:, None, :]
        right = self.right_vectors @ rotations
        right = right * self.sqrt_singular_values[:, None, :]
        return left, right

    def adapted_query_key_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mergeable PyTorch Q/K weights and biases."""
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
        """Compute adapted per-head logits before softmax."""
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
            "side=right, spectrum=full"
        )


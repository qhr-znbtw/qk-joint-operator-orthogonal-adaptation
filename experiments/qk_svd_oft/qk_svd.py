"""Exact compact SVD for the low-rank per-head QK operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from experiments.qk_svd_oft.head_decomposition import augmented_qk_factors


@dataclass(frozen=True)
class QKCompactSVD:
    """Batched compact SVD of all heads in one attention layer."""

    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors_transpose: torch.Tensor

    def reconstruct(self) -> torch.Tensor:
        """Materialize ``U diag(s) V^T`` for diagnostics only."""
        return (
            self.left_vectors
            @ torch.diag_embed(self.singular_values)
            @ self.right_vectors_transpose
        )


def compact_product_svd(
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
) -> QKCompactSVD:
    """Compute the exact compact SVD of ``A B^T`` without forming it.

    ``A`` and ``B`` may be batched and must end in ``[ambient_dim, rank]``.
    Thin QR decompositions reduce the expensive SVD to a batched
    ``[rank, rank]`` problem.
    """
    if left_factors.ndim < 2 or right_factors.ndim < 2:
        raise ValueError("Factors must have at least two dimensions.")
    if left_factors.shape != right_factors.shape:
        raise ValueError("Left and right factors must have identical shapes.")
    if left_factors.shape[-2] < left_factors.shape[-1]:
        raise ValueError("Compact SVD expects ambient_dim >= rank.")

    left_basis, left_triangular = torch.linalg.qr(left_factors, mode="reduced")
    right_basis, right_triangular = torch.linalg.qr(
        right_factors,
        mode="reduced",
    )
    core = left_triangular @ right_triangular.transpose(-1, -2)
    core_left, singular_values, core_right_transpose = torch.linalg.svd(
        core,
        full_matrices=False,
    )

    left_vectors = left_basis @ core_left
    right_vectors_transpose = (
        core_right_transpose @ right_basis.transpose(-1, -2)
    )
    return QKCompactSVD(
        left_vectors=left_vectors,
        singular_values=singular_values,
        right_vectors_transpose=right_vectors_transpose,
    )


def decompose_attention_qk(
    attention: nn.Module,
) -> QKCompactSVD:
    """Decompose every head of a compatible ViT attention module."""
    required = ("query", "key", "num_heads")
    if any(not hasattr(attention, name) for name in required):
        raise TypeError("Attention must expose query, key, and num_heads.")
    query_factors, key_factors = augmented_qk_factors(
        attention.query,
        attention.key,
        attention.num_heads,
    )
    return compact_product_svd(query_factors, key_factors)


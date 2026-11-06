"""Utilities for exposing the per-head query-key bilinear operators.

PyTorch stores an ``nn.Linear`` weight as ``[out_features, in_features]``
and evaluates ``x @ weight.T + bias``.  Consequently, attention heads are
contiguous row blocks of the query and key weights.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def split_linear_heads(
    projection: nn.Linear,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return head weights ``[H, d_h, d_in]`` and biases ``[H, d_h]``."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    if projection.out_features % num_heads != 0:
        raise ValueError(
            f"out_features={projection.out_features} is not divisible by "
            f"num_heads={num_heads}."
        )

    head_size = projection.out_features // num_heads
    weights = projection.weight.reshape(
        num_heads,
        head_size,
        projection.in_features,
    )
    biases = None
    if projection.bias is not None:
        biases = projection.bias.reshape(num_heads, head_size)
    return weights, biases


def project_by_head(
    tokens: torch.Tensor,
    head_weights: torch.Tensor,
    head_biases: torch.Tensor | None,
) -> torch.Tensor:
    """Apply explicit head weights and return ``[B, H, N, d_h]``."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, hidden_size].")
    if head_weights.ndim != 3:
        raise ValueError("head_weights must have shape [heads, head_size, input].")
    if tokens.shape[-1] != head_weights.shape[-1]:
        raise ValueError("Token width and projection input width do not match.")

    projected = torch.einsum("bnd,hsd->bhns", tokens, head_weights)
    if head_biases is not None:
        projected = projected + head_biases[None, :, None, :]
    return projected


def split_projection_attention_logits(
    tokens: torch.Tensor,
    query: nn.Linear,
    key: nn.Linear,
    num_heads: int,
) -> torch.Tensor:
    """Reconstruct scaled attention logits from explicit Q/K head slices."""
    query_weights, query_biases = split_linear_heads(query, num_heads)
    key_weights, key_biases = split_linear_heads(key, num_heads)
    if query_weights.shape != key_weights.shape:
        raise ValueError("Query and key head shapes must match.")

    query_heads = project_by_head(tokens, query_weights, query_biases)
    key_heads = project_by_head(tokens, key_weights, key_biases)
    head_size = query_weights.shape[1]
    return (query_heads @ key_heads.transpose(-2, -1)) / math.sqrt(head_size)


def augmented_qk_factors(
    query: nn.Linear,
    key: nn.Linear,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bias-aware Q/K factors, each shaped ``[H, d+1, d_h]``.

    For augmented tokens ``x_tilde = [x, 1]``, each returned matrix satisfies

    ``q_h = x_tilde @ A_h`` and ``k_h = x_tilde @ B_h``.
    """
    query_weights, query_biases = split_linear_heads(query, num_heads)
    key_weights, key_biases = split_linear_heads(key, num_heads)
    if query_weights.shape != key_weights.shape:
        raise ValueError("Query and key head shapes must match.")
    if query.in_features != key.in_features:
        raise ValueError("Query and key input widths must match.")

    heads, head_size, _ = query_weights.shape
    if query_biases is None:
        query_biases = query_weights.new_zeros(heads, head_size)
    if key_biases is None:
        key_biases = key_weights.new_zeros(heads, head_size)

    # Stored head weights are [d_h, d].  Transpose them to the mathematical
    # column convention [d, d_h], then append each bias as the final input row.
    query_factors = torch.cat(
        (query_weights.transpose(-1, -2), query_biases[:, None, :]),
        dim=1,
    )
    key_factors = torch.cat(
        (key_weights.transpose(-1, -2), key_biases[:, None, :]),
        dim=1,
    )
    return query_factors, key_factors


def augmented_qk_operators(
    query: nn.Linear,
    key: nn.Linear,
    num_heads: int,
) -> torch.Tensor:
    """Build one bias-aware QK operator ``[H, d+1, d+1]`` per head.

    For augmented tokens ``x_tilde = [x, 1]``, each returned matrix satisfies

    ``q_h @ k_h.T = x_tilde @ M_h @ x_tilde.T``.
    """
    query_factors, key_factors = augmented_qk_factors(query, key, num_heads)
    return query_factors @ key_factors.transpose(-1, -2)


def operator_attention_logits(
    tokens: torch.Tensor,
    operators: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Evaluate scaled logits from bias-aware per-head QK operators."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, hidden_size].")
    if operators.ndim != 3:
        raise ValueError("operators must have shape [heads, d+1, d+1].")
    if operators.shape[-2:] != (tokens.shape[-1] + 1,) * 2:
        raise ValueError("Operator width must equal token width plus one.")
    if head_size <= 0:
        raise ValueError("head_size must be positive.")

    ones = tokens.new_ones(*tokens.shape[:-1], 1)
    augmented_tokens = torch.cat((tokens, ones), dim=-1)
    logits = torch.einsum(
        "bni,hij,bmj->bhnm",
        augmented_tokens,
        operators,
        augmented_tokens,
    )
    return logits / math.sqrt(head_size)


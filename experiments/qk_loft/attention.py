"""Low-rank orthogonal support transforms for one joint QK attention layer."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.qk_svd_oft.qk_svd import decompose_attention_qk
from vit_core.vit_backbone import MultiHeadSelfAttention


class QKCentralCalibrationAttention(nn.Module):
    """Expose the exact compact 64x64 QK core for one-time calibration."""

    def __init__(self, attention: MultiHeadSelfAttention) -> None:
        super().__init__()
        if not isinstance(attention, MultiHeadSelfAttention):
            raise TypeError("attention must be MultiHeadSelfAttention.")
        self.num_heads = attention.num_heads
        self.head_size = attention.head_size
        self.hidden_size = attention.query.in_features
        self.query = attention.query
        self.key = attention.key
        self.value = attention.value
        self.output = attention.output

        with torch.no_grad():
            decomposition = decompose_attention_qk(attention)
        self.register_buffer(
            "left_vectors", decomposition.left_vectors.detach().clone()
        )
        self.register_buffer(
            "right_vectors",
            decomposition.right_vectors_transpose.transpose(-1, -2)
            .detach()
            .clone(),
        )
        self.register_buffer(
            "base_singular_values",
            decomposition.singular_values.detach().clone(),
        )
        self.central_operators = nn.Parameter(
            torch.diag_embed(self.base_singular_values).clone()
        )

    def attention_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        ones = tokens.new_ones(*tokens.shape[:-1], 1)
        augmented = torch.cat((tokens, ones), dim=-1)
        left_vectors = self.left_vectors.to(dtype=tokens.dtype)
        right_vectors = self.right_vectors.to(dtype=tokens.dtype)
        core = self.central_operators.to(dtype=tokens.dtype)
        left_coordinates = torch.einsum("bnd,hdr->bhnr", augmented, left_vectors)
        right_coordinates = torch.einsum(
            "bnd,hdr->bhnr", augmented, right_vectors
        )
        logits = torch.einsum(
            "bhnr,hrs,bhms->bhnm",
            left_coordinates,
            core,
            right_coordinates,
        )
        return logits / math.sqrt(self.head_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        return tensor.reshape(
            batch_size, num_tokens, self.num_heads, self.head_size
        ).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_logits(tokens).softmax(dim=-1)
        value = self._split_heads(self.value(tokens))
        context = attention_weights @ value
        return self.output(context.transpose(1, 2).contiguous().flatten(2))


class QKLOFTAttention(nn.Module):
    """Apply fixed-support Cayley rotations to either or both QK subspaces."""

    def __init__(
        self,
        attention: nn.Module,
        *,
        side: str,
        left_support: torch.Tensor | None,
        right_support: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if side not in {"left", "right", "bilateral"}:
            raise ValueError(f"Unknown side: {side}")
        required = ("query", "key", "value", "output", "num_heads", "head_size")
        if any(not hasattr(attention, name) for name in required):
            raise TypeError("attention does not expose the required ViT interface.")

        self.side = side
        self.num_heads = attention.num_heads
        self.head_size = attention.head_size
        self.hidden_size = attention.query.in_features
        self.query = attention.query
        self.key = attention.key
        self.value = attention.value
        self.output = attention.output

        with torch.no_grad():
            decomposition = decompose_attention_qk(attention)
        self.register_buffer(
            "left_vectors",
            decomposition.left_vectors.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_singular_values",
            decomposition.singular_values.detach().clamp_min(0).sqrt(),
            persistent=False,
        )
        self.register_buffer(
            "right_vectors",
            decomposition.right_vectors_transpose.transpose(-1, -2)
            .detach()
            .clone(),
            persistent=False,
        )

        active_left = side in {"left", "bilateral"}
        active_right = side in {"right", "bilateral"}
        self.rank = self._validate_support(left_support, active_left, "left")
        right_rank = self._validate_support(
            right_support, active_right, "right"
        )
        if self.rank is None:
            self.rank = right_rank
        if right_rank is not None and right_rank != self.rank:
            raise ValueError("Left and right support ranks must match.")
        if self.rank is None:
            raise RuntimeError("At least one support must be active.")

        self.register_buffer(
            "left_support",
            left_support.detach().clone() if active_left else None,
        )
        self.register_buffer(
            "right_support",
            right_support.detach().clone() if active_right else None,
        )
        entries = self.rank * (self.rank - 1) // 2
        self.left_skew_values = (
            nn.Parameter(torch.zeros(self.num_heads, entries))
            if active_left
            else None
        )
        self.right_skew_values = (
            nn.Parameter(torch.zeros(self.num_heads, entries))
            if active_right
            else None
        )
        rows, columns = torch.triu_indices(self.rank, self.rank, offset=1)
        self.register_buffer("skew_rows", rows, persistent=False)
        self.register_buffer("skew_columns", columns, persistent=False)
        self._left_transform_cache: torch.Tensor | None = None
        self._right_transform_cache: torch.Tensor | None = None

    def _validate_support(
        self,
        support: torch.Tensor | None,
        active: bool,
        name: str,
    ) -> int | None:
        if not active:
            if support is not None:
                raise ValueError(f"{name}_support must be None when inactive.")
            return None
        if support is None or support.ndim != 3:
            raise ValueError(f"{name}_support must have shape [heads, rank, 64].")
        if support.shape[0] != self.num_heads or support.shape[2] != self.head_size:
            raise ValueError(f"Invalid {name}_support shape: {tuple(support.shape)}")
        gram = support.float() @ support.float().transpose(-1, -2)
        identity = torch.eye(support.shape[1], device=gram.device).expand_as(gram)
        if not torch.allclose(gram, identity, atol=4e-4, rtol=4e-4):
            raise ValueError(f"{name}_support rows are not orthonormal.")
        return support.shape[1]

    def _skew_matrices(self, parameters: torch.Tensor) -> torch.Tensor:
        skew = parameters.new_zeros(self.num_heads, self.rank, self.rank)
        skew[:, self.skew_rows, self.skew_columns] = parameters
        return skew - skew.transpose(-1, -2)

    def left_skew_matrices(self) -> torch.Tensor:
        if self.left_skew_values is None:
            raise RuntimeError("Left transform is inactive.")
        return self._skew_matrices(self.left_skew_values)

    def right_skew_matrices(self) -> torch.Tensor:
        if self.right_skew_values is None:
            raise RuntimeError("Right transform is inactive.")
        return self._skew_matrices(self.right_skew_values)

    def _cayley(self, skew: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=skew.device.type, enabled=False):
            skew = skew.float()
            identity = torch.eye(self.rank, device=skew.device, dtype=skew.dtype)
            return torch.linalg.solve(identity + skew, identity - skew)

    def left_transforms(self) -> torch.Tensor:
        if self._left_transform_cache is not None:
            return self._left_transform_cache
        return self._cayley(self.left_skew_matrices())

    def right_transforms(self) -> torch.Tensor:
        if self._right_transform_cache is not None:
            return self._right_transform_cache
        return self._cayley(self.right_skew_matrices())

    def set_transform_cache(
        self,
        left: torch.Tensor | None,
        right: torch.Tensor | None,
    ) -> None:
        expected = (self.num_heads, self.rank, self.rank)
        if left is not None and left.shape != expected:
            raise ValueError(f"Invalid left cache shape: {tuple(left.shape)}")
        if right is not None and right.shape != expected:
            raise ValueError(f"Invalid right cache shape: {tuple(right.shape)}")
        self._left_transform_cache = left
        self._right_transform_cache = right

    def clear_transform_cache(self) -> None:
        self._left_transform_cache = None
        self._right_transform_cache = None

    @staticmethod
    def _apply_support_transform(
        features: torch.Tensor,
        support: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        support = support.to(dtype=features.dtype)
        transform = transform.to(dtype=features.dtype)
        projected = torch.einsum("hnd,hrd->hnr", features, support)
        transformed = torch.einsum("hnr,hrs->hns", projected, transform)
        correction = torch.einsum(
            "hnr,hrd->hnd", transformed - projected, support
        )
        return features + correction

    def adapted_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.left_vectors
        right = self.right_vectors
        if self.left_support is not None:
            left = self._apply_support_transform(
                left, self.left_support, self.left_transforms()
            )
        if self.right_support is not None:
            right = self._apply_support_transform(
                right, self.right_support, self.right_transforms()
            )
        left = left * self.sqrt_singular_values[:, None, :]
        right = right * self.sqrt_singular_values[:, None, :]
        return left, right

    def dense_embedded_transform(self, side: str) -> torch.Tensor:
        if side == "left":
            support, transform = self.left_support, self.left_transforms()
        elif side == "right":
            support, transform = self.right_support, self.right_transforms()
        else:
            raise ValueError("side must be left or right.")
        if support is None:
            raise RuntimeError(f"{side} transform is inactive.")
        identity = torch.eye(
            self.head_size,
            device=support.device,
            dtype=transform.dtype,
        ).expand(self.num_heads, -1, -1)
        projected_identity = torch.eye(
            self.rank,
            device=transform.device,
            dtype=transform.dtype,
        )
        return identity + support.transpose(-1, -2) @ (
            transform - projected_identity
        ) @ support

    def adapted_query_key_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = self.adapted_factors()
        query_weight = left[:, : self.hidden_size].transpose(-1, -2).reshape(
            self.num_heads * self.head_size, self.hidden_size
        )
        query_bias = left[:, self.hidden_size].reshape(-1)
        key_weight = right[:, : self.hidden_size].transpose(-1, -2).reshape(
            self.num_heads * self.head_size, self.hidden_size
        )
        key_bias = right[:, self.hidden_size].reshape(-1)
        return query_weight, query_bias, key_weight, key_bias

    def attention_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        ones = tokens.new_ones(*tokens.shape[:-1], 1)
        augmented = torch.cat((tokens, ones), dim=-1)
        left, right = self.adapted_factors()
        query = torch.einsum(
            "bnd,hdr->bhnr", augmented, left.to(dtype=tokens.dtype)
        )
        key = torch.einsum(
            "bnd,hdr->bhnr", augmented, right.to(dtype=tokens.dtype)
        )
        return (query @ key.transpose(-1, -2)) / math.sqrt(self.head_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        return tensor.reshape(
            batch_size, num_tokens, self.num_heads, self.head_size
        ).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_logits(tokens).softmax(dim=-1)
        value = self._split_heads(self.value(tokens))
        context = attention_weights @ value
        return self.output(context.transpose(1, 2).contiguous().flatten(2))

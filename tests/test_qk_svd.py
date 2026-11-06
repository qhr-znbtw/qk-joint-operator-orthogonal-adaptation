"""Tests for the exact low-rank QK singular value decomposition."""

import torch

from experiments.qk_svd_oft.head_decomposition import (
    augmented_qk_factors,
    augmented_qk_operators,
)
from experiments.qk_svd_oft.qk_svd import (
    compact_product_svd,
    decompose_attention_qk,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import (
    MultiHeadSelfAttention,
    ViTConfig,
    ViTForImageClassification,
)


def assert_svd_is_exact(attention: MultiHeadSelfAttention) -> float:
    query_factors, key_factors = augmented_qk_factors(
        attention.query,
        attention.key,
        attention.num_heads,
    )
    expected = query_factors @ key_factors.transpose(-1, -2)
    decomposition = compact_product_svd(query_factors, key_factors)
    actual = decomposition.reconstruct()

    heads = attention.num_heads
    ambient_size = attention.query.in_features + 1
    head_size = attention.query.out_features // heads
    assert decomposition.left_vectors.shape == (
        heads,
        ambient_size,
        head_size,
    )
    assert decomposition.singular_values.shape == (heads, head_size)
    assert decomposition.right_vectors_transpose.shape == (
        heads,
        head_size,
        ambient_size,
    )

    identity = torch.eye(
        head_size,
        device=actual.device,
        dtype=actual.dtype,
    ).expand(heads, -1, -1)
    torch.testing.assert_close(
        decomposition.left_vectors.transpose(-1, -2)
        @ decomposition.left_vectors,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        decomposition.right_vectors_transpose
        @ decomposition.right_vectors_transpose.transpose(-1, -2),
        identity,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    return (actual - expected).abs().max().item()


def test_small_qk_compact_svd() -> None:
    torch.manual_seed(17)
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        hidden_size=12,
        num_heads=3,
        intermediate_size=24,
        num_layers=1,
    )
    attention = MultiHeadSelfAttention(config)
    assert_svd_is_exact(attention)


def test_pretrained_vit_all_qk_compact_svds() -> None:
    model = ViTForImageClassification(num_classes=3)
    assert load_pretrained_vit(model) == 198

    errors = []
    with torch.no_grad():
        for block in model.encoder.blocks:
            decomposition = decompose_attention_qk(block.attention)
            expected = augmented_qk_operators(
                block.attention.query,
                block.attention.key,
                block.attention.num_heads,
            )
            actual = decomposition.reconstruct()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            errors.append((actual - expected).abs().max().item())

    print(f"Pretrained 144-head compact-SVD maximum error: {max(errors):.3e}")


if __name__ == "__main__":
    test_small_qk_compact_svd()
    test_pretrained_vit_all_qk_compact_svds()
    print("Compact per-head QK SVD tests passed.")


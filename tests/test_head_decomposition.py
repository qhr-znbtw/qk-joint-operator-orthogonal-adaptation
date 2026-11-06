"""Verify that per-head Q/K matrices reproduce the original ViT logits."""

import math

import torch

from experiments.qk_svd_oft.head_decomposition import (
    augmented_qk_operators,
    operator_attention_logits,
    split_linear_heads,
    split_projection_attention_logits,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import (
    MultiHeadSelfAttention,
    ViTConfig,
    ViTForImageClassification,
)


def model_attention_logits(
    attention: MultiHeadSelfAttention,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Use the model's own projection and reshape path as the reference."""
    query = attention._split_heads(attention.query(tokens))
    key = attention._split_heads(attention.key(tokens))
    return (query @ key.transpose(-2, -1)) / math.sqrt(attention.head_size)


def assert_reconstruction(
    attention: MultiHeadSelfAttention,
    tokens: torch.Tensor,
) -> tuple[float, float]:
    """Check both explicit head slicing and the augmented QK operator."""
    reference = model_attention_logits(attention, tokens)
    split_logits = split_projection_attention_logits(
        tokens,
        attention.query,
        attention.key,
        attention.num_heads,
    )
    operators = augmented_qk_operators(
        attention.query,
        attention.key,
        attention.num_heads,
    )
    operator_logits = operator_attention_logits(
        tokens,
        operators,
        attention.head_size,
    )

    split_error = (reference - split_logits).abs().max().item()
    operator_error = (reference - operator_logits).abs().max().item()
    torch.testing.assert_close(split_logits, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(operator_logits, reference, atol=2e-5, rtol=2e-5)
    return split_error, operator_error


def test_small_attention_head_reconstruction() -> None:
    torch.manual_seed(11)
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        hidden_size=12,
        num_heads=3,
        intermediate_size=24,
        num_layers=1,
    )
    attention = MultiHeadSelfAttention(config)
    tokens = torch.randn(2, 5, config.hidden_size)

    query_heads, query_biases = split_linear_heads(
        attention.query,
        config.num_heads,
    )
    assert query_heads.shape == (3, 4, 12)
    assert query_biases is not None and query_biases.shape == (3, 4)
    assert augmented_qk_operators(
        attention.query,
        attention.key,
        config.num_heads,
    ).shape == (3, 13, 13)
    assert_reconstruction(attention, tokens)


def test_pretrained_vit_all_blocks_reconstruction() -> None:
    torch.manual_seed(13)
    model = ViTForImageClassification(num_classes=3)
    assert load_pretrained_vit(model) == 198
    model.eval()

    with torch.no_grad():
        images = torch.randn(1, 3, 224, 224)
        tokens = model.embedding(images)
        split_errors = []
        operator_errors = []
        for block in model.encoder.blocks:
            attention_tokens = block.norm1(tokens)
            split_error, operator_error = assert_reconstruction(
                block.attention,
                attention_tokens,
            )
            split_errors.append(split_error)
            operator_errors.append(operator_error)
            tokens = block(tokens)

    print(
        "Pretrained 12-block maximum errors: "
        f"split={max(split_errors):.3e}, "
        f"augmented_operator={max(operator_errors):.3e}"
    )


if __name__ == "__main__":
    test_small_attention_head_reconstruction()
    test_pretrained_vit_all_blocks_reconstruction()
    print("Per-head QK reconstruction tests passed.")

"""Correctness tests for right-sided full-spectrum joint-QK BOFT."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_boft_common import (
    build_optimizer,
    count_parameters,
    single_side_diagnostics,
)
from experiments.qk_boft_right.attention import QKBOFTRightAttention
from experiments.qk_boft_right.model import ViTQKBOFTRightClassifier
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import MultiHeadSelfAttention, ViTConfig


def tiny_attention() -> tuple[MultiHeadSelfAttention, ViTConfig]:
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        num_layers=1,
    )
    return MultiHeadSelfAttention(config), config


def regular_logits(attention, tokens):
    query = attention._split_heads(attention.query(tokens))
    key = attention._split_heads(attention.key(tokens))
    return (query @ key.transpose(-2, -1)) / math.sqrt(attention.head_size)


def test_right_formula_and_merge() -> None:
    torch.manual_seed(67)
    attention, config = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKBOFTRightAttention(attention, block_size=4, num_factors=2)
    tokens = torch.randn(2, 5, config.hidden_size)
    torch.testing.assert_close(adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5)

    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.15)
    rotation = adapted.dense_orthogonal_matrices()
    singular_values = adapted.sqrt_singular_values.square()
    expected = (
        adapted.left_vectors
        @ torch.diag_embed(singular_values)
        @ rotation.transpose(-1, -2)
        @ adapted.right_vectors.transpose(-1, -2)
    )
    left, right = adapted.adapted_factors()
    torch.testing.assert_close(left @ right.transpose(-1, -2), expected, atol=3e-5, rtol=3e-5)

    qw, qb, kw, kb = adapted.adapted_query_key_parameters()
    merged = MultiHeadSelfAttention(config)
    with torch.no_grad():
        merged.query.weight.copy_(qw)
        merged.query.bias.copy_(qb)
        merged.key.weight.copy_(kw)
        merged.key.bias.copy_(kb)
    torch.testing.assert_close(adapted.attention_logits(tokens), regular_logits(merged, tokens), atol=3e-5, rtol=3e-5)


def test_pretrained_right_model() -> None:
    torch.manual_seed(71)
    model = ViTQKBOFTRightClassifier(num_classes=5)
    assert load_pretrained_vit(model) == 198
    model.eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = model(images)
    model.initialize_adapters()
    model.eval()
    with torch.no_grad():
        actual = model(images)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    assert count_parameters(model, 69_120)["boft"] == 69_120

    optimizer = build_optimizer(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), torch.tensor([2])).backward()
    diagnostics = single_side_diagnostics(model)
    assert diagnostics["rotation_gradient_norm"] > 0
    print(
        "Right QK-BOFT diagnostics: "
        f"initialization_error={(actual - expected).abs().max().item():.3e}, "
        f"gradient_norm={diagnostics['rotation_gradient_norm']:.3e}"
    )


if __name__ == "__main__":
    test_right_formula_and_merge()
    test_pretrained_right_model()
    print("QK-BOFT-Full-Right tests passed.")

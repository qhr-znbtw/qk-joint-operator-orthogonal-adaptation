"""Tests for bilateral QK-BOFT with unconstrained spectrum learning."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_boft_bilateral_spectrum.attention import (
    QKBOFTBilateralSpectrumAttention,
)
from experiments.qk_boft_bilateral_spectrum.model import (
    ViTQKBOFTBilateralSpectrumClassifier,
)
from experiments.qk_boft_bilateral_spectrum.run_qk_boft_bilateral_spectrum import (
    spectrum_diagnostics,
)
from experiments.qk_boft_common import build_optimizer, count_parameters
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


def test_spectrum_initialization_and_global_scaling() -> None:
    torch.manual_seed(83)
    attention, _ = tiny_attention()
    adapted = QKBOFTBilateralSpectrumAttention(
        attention, block_size=4, num_factors=2
    )
    base = adapted.base_singular_values()
    learned = adapted.learned_singular_values()
    torch.testing.assert_close(learned, base, atol=1e-7, rtol=1e-7)

    with torch.no_grad():
        adapted.spectrum_log_scales.normal_(mean=0.0, std=0.4)
    learned = adapted.learned_singular_values()
    assert torch.all(learned > 0)
    before_shift = learned.clone()
    with torch.no_grad():
        adapted.spectrum_log_scales.add_(0.7)
    torch.testing.assert_close(
        adapted.learned_singular_values(),
        before_shift * torch.exp(torch.tensor(0.7)),
        atol=2e-6,
        rtol=2e-6,
    )


def test_spectral_operator_formula_and_merge() -> None:
    torch.manual_seed(89)
    attention, config = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKBOFTBilateralSpectrumAttention(
        attention, block_size=4, num_factors=2
    )
    tokens = torch.randn(2, 5, config.hidden_size)
    torch.testing.assert_close(
        adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5
    )

    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.15)
        adapted.right_skew_parameters.normal_(mean=0.0, std=0.15)
        adapted.spectrum_log_scales.normal_(mean=0.0, std=0.2)
    left_rotation = adapted.dense_left_orthogonal_matrices()
    right_rotation = adapted.dense_right_orthogonal_matrices()
    singular_values = adapted.learned_singular_values()
    expected = (
        adapted.left_vectors
        @ left_rotation
        @ torch.diag_embed(singular_values)
        @ right_rotation.transpose(-1, -2)
        @ adapted.right_vectors.transpose(-1, -2)
    )
    left, right = adapted.adapted_factors()
    torch.testing.assert_close(
        left @ right.transpose(-1, -2), expected, atol=3e-5, rtol=3e-5
    )

    qw, qb, kw, kb = adapted.adapted_query_key_parameters()
    merged = MultiHeadSelfAttention(config)
    with torch.no_grad():
        merged.query.weight.copy_(qw)
        merged.query.bias.copy_(qb)
        merged.key.weight.copy_(kw)
        merged.key.bias.copy_(kb)
    torch.testing.assert_close(
        adapted.attention_logits(tokens),
        regular_logits(merged, tokens),
        atol=3e-5,
        rtol=3e-5,
    )


def test_pretrained_spectral_model_and_gradients() -> None:
    torch.manual_seed(97)
    model = ViTQKBOFTBilateralSpectrumClassifier(num_classes=5)
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
    counts = count_parameters(model, 138_240, 9_216)
    assert counts["spectrum"] == 9_216

    optimizer = build_optimizer(model, spectrum_lr=1e-4)
    assert [group["lr"] for group in optimizer.param_groups] == [
        5e-4,
        1e-4,
        1e-3,
    ]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), torch.tensor([2])).backward()
    diagnostics = spectrum_diagnostics(model)
    assert diagnostics["left_rotation_gradient_norm"] > 0
    assert diagnostics["right_rotation_gradient_norm"] > 0
    assert diagnostics["spectrum_gradient_norm"] > 0
    assert diagnostics["mean_relative_spectrum_change"] == 0.0
    print(
        "Learnable-spectrum diagnostics: "
        f"initialization_error={(actual - expected).abs().max().item():.3e}, "
        f"spectrum_gradient={diagnostics['spectrum_gradient_norm']:.3e}"
    )


if __name__ == "__main__":
    test_spectrum_initialization_and_global_scaling()
    test_spectral_operator_formula_and_merge()
    test_pretrained_spectral_model_and_gradients()
    print("QK-BOFT-Bilateral-Learnable-Spectrum tests passed.")

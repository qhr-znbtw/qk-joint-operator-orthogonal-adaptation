"""Correctness tests for full-spectrum right-sided joint-QK PSOFT."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_psoft_right.attention import QKPSOFTRightAttention
from experiments.qk_psoft_right.model import ViTQKPSOFTRightClassifier
from experiments.qk_psoft_right.run_qk_psoft_right import (
    build_optimizer,
    count_parameters,
    rotation_diagnostics,
)
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import MultiHeadSelfAttention, ViTConfig


def tiny_attention() -> tuple[MultiHeadSelfAttention, ViTConfig]:
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        hidden_size=12,
        num_heads=3,
        intermediate_size=24,
        num_layers=1,
    )
    return MultiHeadSelfAttention(config), config


def regular_logits(
    attention: MultiHeadSelfAttention,
    tokens: torch.Tensor,
) -> torch.Tensor:
    query = attention._split_heads(attention.query(tokens))
    key = attention._split_heads(attention.key(tokens))
    return (query @ key.transpose(-2, -1)) / math.sqrt(attention.head_size)


def test_identity_orthogonality_and_nonzero_gradient() -> None:
    torch.manual_seed(31)
    attention, _ = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKPSOFTRightAttention(attention)
    tokens = torch.randn(2, 5, 12)

    identity = torch.eye(4).expand(3, -1, -1)
    torch.testing.assert_close(adapted.orthogonal_matrices(), identity)
    torch.testing.assert_close(adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5)

    loss = adapted(tokens).square().mean()
    loss.backward()
    assert adapted.skew_parameters.grad is not None
    assert adapted.skew_parameters.grad.norm().item() > 0

    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.1)
    rotations = adapted.orthogonal_matrices()
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )


def test_right_side_formula_and_merge() -> None:
    torch.manual_seed(37)
    attention, config = tiny_attention()
    adapted = QKPSOFTRightAttention(attention)
    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.15)

    rotations = adapted.orthogonal_matrices()
    singular_values = adapted.sqrt_singular_values.square()
    expected_operator = (
        adapted.left_vectors
        @ torch.diag_embed(singular_values)
        @ rotations.transpose(-1, -2)
        @ adapted.right_vectors.transpose(-1, -2)
    )
    left, right = adapted.adapted_factors()
    actual_operator = left @ right.transpose(-1, -2)
    torch.testing.assert_close(
        actual_operator,
        expected_operator,
        atol=2e-5,
        rtol=2e-5,
    )

    query_weight, query_bias, key_weight, key_bias = (
        adapted.adapted_query_key_parameters()
    )
    merged = MultiHeadSelfAttention(config)
    with torch.no_grad():
        merged.query.weight.copy_(query_weight)
        merged.query.bias.copy_(query_bias)
        merged.key.weight.copy_(key_weight)
        merged.key.bias.copy_(key_bias)

    tokens = torch.randn(2, 6, config.hidden_size)
    torch.testing.assert_close(
        adapted.attention_logits(tokens),
        regular_logits(merged, tokens),
        atol=2e-5,
        rtol=2e-5,
    )


def test_pretrained_model_initialization_and_parameter_budget() -> None:
    torch.manual_seed(41)
    model = ViTQKPSOFTRightClassifier(num_classes=5)
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
    initialization_error = (actual - expected).abs().max().item()

    counts = count_parameters(model)
    assert counts["rotation"] == 290_304
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert all(
        name.endswith(".skew_parameters")
        or name.startswith("head.classifier.")
        for name in trainable_names
    )
    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-4, 1e-3]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), torch.tensor([2]))
    loss.backward()
    rotation_gradient_norm = torch.sqrt(
        sum(
            layer.skew_parameters.grad.square().sum()
            for layer in model.adapter_layers()
            if layer.skew_parameters.grad is not None
        )
    ).item()
    assert rotation_gradient_norm > 0
    diagnostics = rotation_diagnostics(model)
    assert diagnostics["rotation_gradient_norm"] > 0
    assert diagnostics["mean_rotation_distance"] == 0.0
    first_rotation_before = model.adapter_layers()[0].skew_parameters.detach().clone()
    optimizer.step()
    assert not torch.equal(
        model.adapter_layers()[0].skew_parameters,
        first_rotation_before,
    )
    print(
        "Pretrained right-side diagnostics: "
        f"initialization_error={initialization_error:.3e}, "
        f"rotation_gradient_norm={rotation_gradient_norm:.3e}"
    )


if __name__ == "__main__":
    test_identity_orthogonality_and_nonzero_gradient()
    test_right_side_formula_and_merge()
    test_pretrained_model_initialization_and_parameter_budget()
    print("QK-PSOFT-Full-Right tests passed.")


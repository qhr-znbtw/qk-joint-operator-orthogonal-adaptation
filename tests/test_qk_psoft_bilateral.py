"""Correctness tests for full-spectrum bilateral joint-QK PSOFT."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_psoft_bilateral.attention import QKPSOFTBilateralAttention
from experiments.qk_psoft_bilateral.model import ViTQKPSOFTBilateralClassifier
from experiments.qk_psoft_bilateral.run_qk_psoft_bilateral import (
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


def test_identity_orthogonality_and_two_nonzero_gradients() -> None:
    torch.manual_seed(43)
    attention, _ = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKPSOFTBilateralAttention(attention)
    tokens = torch.randn(2, 5, 12)

    identity = torch.eye(4).expand(3, -1, -1)
    torch.testing.assert_close(adapted.left_orthogonal_matrices(), identity)
    torch.testing.assert_close(adapted.right_orthogonal_matrices(), identity)
    torch.testing.assert_close(adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5)

    loss = adapted(tokens).square().mean()
    loss.backward()
    assert adapted.left_skew_parameters.grad is not None
    assert adapted.right_skew_parameters.grad is not None
    assert adapted.left_skew_parameters.grad.norm().item() > 0
    assert adapted.right_skew_parameters.grad.norm().item() > 0

    with torch.no_grad():
        adapted.left_skew_parameters.normal_(mean=0.0, std=0.1)
        adapted.right_skew_parameters.normal_(mean=0.0, std=0.1)
    for rotations in (
        adapted.left_orthogonal_matrices(),
        adapted.right_orthogonal_matrices(),
    ):
        torch.testing.assert_close(
            rotations.transpose(-1, -2) @ rotations,
            identity,
            atol=2e-5,
            rtol=2e-5,
        )


def test_bilateral_formula_and_merge() -> None:
    torch.manual_seed(47)
    attention, config = tiny_attention()
    adapted = QKPSOFTBilateralAttention(attention)
    with torch.no_grad():
        adapted.left_skew_parameters.normal_(mean=0.0, std=0.15)
        adapted.right_skew_parameters.normal_(mean=0.0, std=0.15)

    left_rotations = adapted.left_orthogonal_matrices()
    right_rotations = adapted.right_orthogonal_matrices()
    singular_values = adapted.sqrt_singular_values.square()
    expected_operator = (
        adapted.left_vectors
        @ left_rotations
        @ torch.diag_embed(singular_values)
        @ right_rotations.transpose(-1, -2)
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
    torch.manual_seed(53)
    model = ViTQKPSOFTBilateralClassifier(num_classes=5)
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
    assert counts["left_rotation"] == 290_304
    assert counts["right_rotation"] == 290_304
    assert counts["rotation"] == 580_608
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert all(
        name.endswith("_skew_parameters")
        or name.startswith("head.classifier.")
        for name in trainable_names
    )
    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-4, 1e-3]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), torch.tensor([2]))
    loss.backward()
    left_gradient_norm = torch.sqrt(
        sum(
            layer.left_skew_parameters.grad.square().sum()
            for layer in model.adapter_layers()
        )
    ).item()
    right_gradient_norm = torch.sqrt(
        sum(
            layer.right_skew_parameters.grad.square().sum()
            for layer in model.adapter_layers()
        )
    ).item()
    assert left_gradient_norm > 0
    assert right_gradient_norm > 0
    diagnostics = rotation_diagnostics(model)
    assert diagnostics["left_rotation_gradient_norm"] > 0
    assert diagnostics["right_rotation_gradient_norm"] > 0
    assert diagnostics["mean_rotation_distance"] == 0.0

    first_layer = model.adapter_layers()[0]
    left_before = first_layer.left_skew_parameters.detach().clone()
    right_before = first_layer.right_skew_parameters.detach().clone()
    optimizer.step()
    assert not torch.equal(first_layer.left_skew_parameters, left_before)
    assert not torch.equal(first_layer.right_skew_parameters, right_before)
    print(
        "Pretrained bilateral diagnostics: "
        f"initialization_error={initialization_error:.3e}, "
        f"left_gradient_norm={left_gradient_norm:.3e}, "
        f"right_gradient_norm={right_gradient_norm:.3e}"
    )


if __name__ == "__main__":
    test_identity_orthogonality_and_two_nonzero_gradients()
    test_bilateral_formula_and_merge()
    test_pretrained_model_initialization_and_parameter_budget()
    print("QK-PSOFT-Full-Bilateral tests passed.")


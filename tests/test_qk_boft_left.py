"""Correctness tests for full-spectrum left-sided joint-QK BOFT."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_boft_left.attention import (
    QKBOFTLeftAttention,
    build_coordinate_permutation,
)
from experiments.qk_boft_left.model import ViTQKBOFTLeftClassifier
from experiments.qk_boft_left.run_qk_boft_left import (
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
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
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


def test_permutations_and_full_coverage() -> None:
    expected_orders = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 4, 5, 2, 3, 6, 7],
    ]
    for factor_index, expected in enumerate(expected_orders):
        actual = build_coordinate_permutation(8, 4, factor_index)
        assert actual.tolist() == expected

    attention, _ = tiny_attention()
    adapted = QKBOFTLeftAttention(attention, block_size=4, num_factors=2)
    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.2)
    dense = adapted.dense_orthogonal_matrices()
    assert torch.all((dense.abs() > 1e-7).sum(dim=-1) == 8)


def test_identity_orthogonality_formula_and_merge() -> None:
    torch.manual_seed(59)
    attention, config = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKBOFTLeftAttention(attention, block_size=4, num_factors=2)
    tokens = torch.randn(2, 5, config.hidden_size)

    identity = torch.eye(4).expand(2, 2, 2, -1, -1)
    torch.testing.assert_close(adapted.orthogonal_blocks(), identity)
    torch.testing.assert_close(adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5)

    loss = adapted(tokens).square().mean()
    loss.backward()
    assert adapted.skew_parameters.grad is not None
    assert adapted.skew_parameters.grad.norm().item() > 0

    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.15)
    blocks = adapted.orthogonal_blocks()
    torch.testing.assert_close(
        blocks.transpose(-1, -2) @ blocks,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )
    dense = adapted.dense_orthogonal_matrices()
    dense_identity = torch.eye(8).expand(2, -1, -1)
    torch.testing.assert_close(
        dense.transpose(-1, -2) @ dense,
        dense_identity,
        atol=3e-5,
        rtol=3e-5,
    )

    singular_values = adapted.sqrt_singular_values.square()
    expected_operator = (
        adapted.left_vectors
        @ dense
        @ torch.diag_embed(singular_values)
        @ adapted.right_vectors.transpose(-1, -2)
    )
    left, right = adapted.adapted_factors()
    torch.testing.assert_close(
        left @ right.transpose(-1, -2),
        expected_operator,
        atol=3e-5,
        rtol=3e-5,
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
    torch.testing.assert_close(
        adapted.attention_logits(tokens),
        regular_logits(merged, tokens),
        atol=3e-5,
        rtol=3e-5,
    )


def test_pretrained_model_initialization_and_parameter_budget() -> None:
    torch.manual_seed(61)
    model = ViTQKBOFTLeftClassifier(num_classes=5)
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
    assert counts["boft"] == 69_120
    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-4, 1e-3]
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

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), torch.tensor([2]))
    loss.backward()
    diagnostics = rotation_diagnostics(model)
    assert diagnostics["rotation_gradient_norm"] > 0
    assert diagnostics["mean_rotation_distance"] == 0.0
    first_before = model.adapter_layers()[0].skew_parameters.detach().clone()
    optimizer.step()
    assert not torch.equal(model.adapter_layers()[0].skew_parameters, first_before)
    print(
        "Pretrained QK-BOFT diagnostics: "
        f"initialization_error={initialization_error:.3e}, "
        f"gradient_norm={diagnostics['rotation_gradient_norm']:.3e}"
    )


if __name__ == "__main__":
    test_permutations_and_full_coverage()
    test_identity_orthogonality_formula_and_merge()
    test_pretrained_model_initialization_and_parameter_budget()
    print("QK-BOFT-Full-Left tests passed.")


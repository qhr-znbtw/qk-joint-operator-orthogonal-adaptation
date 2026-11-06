"""Formula-level and integration tests for the QK-LOFT factorial study."""

import torch

from experiments.qk_loft.attention import (
    QKCentralCalibrationAttention,
    QKLOFTAttention,
)
from experiments.qk_loft.model import build_qk_loft_model
from experiments.qk_loft.run_qk_loft import count_parameters
from experiments.qk_loft.supports import (
    principal_support,
    supports_from_gradient,
)
from vit_core.vit_backbone import MultiHeadSelfAttention, ViTConfig


def _tiny_attention() -> MultiHeadSelfAttention:
    torch.manual_seed(3)
    config = ViTConfig(
        image_size=4,
        patch_size=2,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        num_layers=1,
    )
    return MultiHeadSelfAttention(config)


def _supports() -> torch.Tensor:
    return principal_support(num_heads=2, width=4, rank=2)


def test_calibration_core_reproduces_original_attention() -> None:
    base = _tiny_attention().eval()
    wrapped = QKCentralCalibrationAttention(base).eval()
    tokens = torch.randn(2, 5, 8)
    assert torch.allclose(wrapped(tokens), base(tokens), atol=2e-5, rtol=2e-5)


def test_support_constructors_are_row_orthonormal() -> None:
    torch.manual_seed(4)
    gradient = torch.randn(3, 6, 6)
    singular_values = torch.rand(3, 6)
    for kind in ("grad_svd", "skew_grad"):
        left, right = supports_from_gradient(
            gradient,
            singular_values,
            support_type=kind,
            rank=4,
        )
        identity = torch.eye(4).expand(3, -1, -1)
        assert torch.allclose(left @ left.transpose(-1, -2), identity, atol=2e-5)
        assert torch.allclose(right @ right.transpose(-1, -2), identity, atol=2e-5)


def test_all_sides_match_dense_embedded_formula_and_merge() -> None:
    tokens = torch.randn(2, 5, 8)
    for side in ("left", "right", "bilateral"):
        base = _tiny_attention()
        adapter = QKLOFTAttention(
            base,
            side=side,
            left_support=_supports() if side in {"left", "bilateral"} else None,
            right_support=_supports() if side in {"right", "bilateral"} else None,
        )
        with torch.no_grad():
            if adapter.left_skew_values is not None:
                adapter.left_skew_values.normal_(std=0.08)
            if adapter.right_skew_values is not None:
                adapter.right_skew_values.normal_(std=0.08)

        left_expected = adapter.left_vectors
        right_expected = adapter.right_vectors
        if adapter.left_support is not None:
            left_expected = left_expected @ adapter.dense_embedded_transform("left")
        if adapter.right_support is not None:
            right_expected = right_expected @ adapter.dense_embedded_transform("right")
        left, right = adapter.adapted_factors()
        sqrt_s = adapter.sqrt_singular_values[:, None, :]
        assert torch.allclose(left, left_expected * sqrt_s, atol=2e-5, rtol=2e-5)
        assert torch.allclose(right, right_expected * sqrt_s, atol=2e-5, rtol=2e-5)

        query_weight, query_bias, key_weight, key_bias = (
            adapter.adapted_query_key_parameters()
        )
        query = torch.nn.functional.linear(tokens, query_weight, query_bias)
        key = torch.nn.functional.linear(tokens, key_weight, key_bias)
        query = query.reshape(2, 5, 2, 4).transpose(1, 2)
        key = key.reshape(2, 5, 2, 4).transpose(1, 2)
        merged_logits = query @ key.transpose(-1, -2) / 2.0
        assert torch.allclose(
            adapter.attention_logits(tokens),
            merged_logits,
            atol=3e-5,
            rtol=3e-5,
        )


def test_full_principal_models_have_controlled_parameter_counts() -> None:
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    for side, expected in (
        ("left", 149_040),
        ("right", 149_040),
        ("bilateral", 298_080),
    ):
        model, metadata = build_qk_loft_model(
            10,
            "principal",
            side,
            device=device,
            criterion=criterion,
            calibration_loader=None,
            calibration_batches=1,
        )
        counts = count_parameters(model)
        assert counts["adapter"] == expected
        assert metadata["num_heads_adapted"] == 144


def test_active_qk_loft_parameters_receive_gradients() -> None:
    adapter = QKLOFTAttention(
        _tiny_attention(),
        side="bilateral",
        left_support=_supports(),
        right_support=_supports(),
    )
    loss = adapter(torch.randn(2, 5, 8)).square().mean()
    loss.backward()
    assert adapter.left_skew_values.grad is not None
    assert adapter.right_skew_values.grad is not None
    assert adapter.left_skew_values.grad.norm() > 0
    assert adapter.right_skew_values.grad.norm() > 0

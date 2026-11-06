"""Correctness tests for internal-external bilateral joint-QK BOFT."""

import copy
import math

import torch
from torch.nn import functional as F

from experiments.qk_boft_common import build_optimizer, count_parameters
from experiments.qk_boft_internal_external.attention import (
    QKBOFTInternalExternalAttention,
)
from experiments.qk_boft_internal_external.config import TOTAL_BOFT_PARAMETERS
from experiments.qk_boft_internal_external.diagnostics import (
    internal_external_diagnostics,
)
from experiments.qk_boft_internal_external.external_boft import (
    GlobalBilateralExternalBOFT,
)
from experiments.qk_boft_internal_external.model import (
    ViTQKBOFTInternalExternalClassifier,
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


def regular_logits(attention, tokens):
    query = attention._split_heads(attention.query(tokens))
    key = attention._split_heads(attention.key(tokens))
    return (query @ key.transpose(-2, -1)) / math.sqrt(attention.head_size)


def augmented_rotation(row_rotation: torch.Tensor) -> torch.Tensor:
    dimension = row_rotation.shape[0]
    output = torch.eye(dimension + 1, dtype=row_rotation.dtype)
    output[:dimension, :dimension] = row_rotation.transpose(0, 1)
    return output


def test_internal_external_formula_and_identity() -> None:
    torch.manual_seed(101)
    attention, config = tiny_attention()
    reference = copy.deepcopy(attention)
    adapted = QKBOFTInternalExternalAttention(
        attention,
        block_size=4,
        num_factors=2,
        external_block_size=4,
        external_num_factors=3,
    )
    external = GlobalBilateralExternalBOFT(
        dimension=16,
        block_size=4,
        num_factors=3,
    )
    tokens = torch.randn(2, 5, config.hidden_size)
    adapted.set_external_rotation_cache(
        external.left_orthogonal_blocks(),
        external.right_orthogonal_blocks(),
    )
    torch.testing.assert_close(
        adapted(tokens), reference(tokens), atol=2e-5, rtol=2e-5
    )

    with torch.no_grad():
        adapted.skew_parameters.normal_(mean=0.0, std=0.12)
        adapted.right_skew_parameters.normal_(mean=0.0, std=0.12)
        external.left_skew_parameters.normal_(mean=0.0, std=0.08)
        external.right_skew_parameters.normal_(mean=0.0, std=0.08)
    adapted.set_external_rotation_cache(
        external.left_orthogonal_blocks(),
        external.right_orthogonal_blocks(),
    )

    internal = (
        adapted.left_vectors
        @ adapted.dense_left_orthogonal_matrices()
        @ torch.diag_embed(adapted.sqrt_singular_values.square())
        @ adapted.dense_right_orthogonal_matrices().transpose(-1, -2)
        @ adapted.right_vectors.transpose(-1, -2)
    )
    q_left = augmented_rotation(external.dense_left_row_rotation())
    q_right = augmented_rotation(external.dense_right_row_rotation())
    expected = q_left @ internal @ q_right.transpose(-1, -2)
    left, right = adapted.adapted_factors()
    torch.testing.assert_close(
        left @ right.transpose(-1, -2), expected, atol=4e-5, rtol=4e-5
    )


def test_pretrained_internal_external_model() -> None:
    torch.manual_seed(103)
    model = ViTQKBOFTInternalExternalClassifier(num_classes=5)
    assert load_pretrained_vit(model) == 198
    model.eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = model(images)
    model.initialize_adapters()
    model.eval()
    with torch.no_grad():
        actual = model(images)
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)
    counts = count_parameters(model, TOTAL_BOFT_PARAMETERS)
    assert counts["boft"] == TOTAL_BOFT_PARAMETERS

    optimizer = build_optimizer(model, external_boft_lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), torch.tensor([2])).backward()
    diagnostics = internal_external_diagnostics(model)
    assert diagnostics["left_rotation_gradient_norm"] > 0
    assert diagnostics["right_rotation_gradient_norm"] > 0
    assert diagnostics["external_left_rotation_gradient_norm"] > 0
    assert diagnostics["external_right_rotation_gradient_norm"] > 0
    print(
        "Internal-external diagnostics: "
        f"initialization_error={(actual - expected).abs().max().item():.3e}, "
        f"external_left_gradient="
        f"{diagnostics['external_left_rotation_gradient_norm']:.3e}, "
        f"external_right_gradient="
        f"{diagnostics['external_right_rotation_gradient_norm']:.3e}"
    )


if __name__ == "__main__":
    test_internal_external_formula_and_identity()
    test_pretrained_internal_external_model()
    print("QK-BOFT internal-external tests passed.")

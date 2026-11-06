"""Tests for layer-wise external plus head-wise internal QK-BOFT."""

import torch
from torch.nn import functional as F

from experiments.qk_boft_common import build_optimizer, count_parameters
from experiments.qk_boft_layerwise_internal_external.config import (
    TOTAL_BOFT_PARAMETERS,
)
from experiments.qk_boft_layerwise_internal_external.diagnostics import (
    layerwise_internal_external_diagnostics,
)
from experiments.qk_boft_layerwise_internal_external.external_boft import (
    LayerwiseBilateralExternalBOFT,
)
from experiments.qk_boft_layerwise_internal_external.model import (
    ViTQKBOFTLayerwiseInternalExternalClassifier,
)
from vit_core.pretrained import load_pretrained_vit


def test_layerwise_external_identity_and_independence() -> None:
    torch.manual_seed(107)
    adapter = LayerwiseBilateralExternalBOFT(
        num_layers=2,
        dimension=16,
        block_size=4,
        num_factors=3,
    )
    factors = torch.randn(2, 2, 17, 8)
    identity_output = adapter.rotate_feature_rows(
        factors,
        adapter.left_orthogonal_blocks(),
    )
    torch.testing.assert_close(identity_output, factors, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        adapter.left_skew_parameters[0].normal_(mean=0.0, std=0.1)
    changed = adapter.rotate_feature_rows(
        factors,
        adapter.left_orthogonal_blocks(),
    )
    assert not torch.allclose(changed[0, :, :-1], factors[0, :, :-1])
    torch.testing.assert_close(changed[1], factors[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(changed[..., -1, :], factors[..., -1, :])


def test_pretrained_layerwise_internal_external_model() -> None:
    torch.manual_seed(109)
    model = ViTQKBOFTLayerwiseInternalExternalClassifier(num_classes=5)
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
    assert count_parameters(model, TOTAL_BOFT_PARAMETERS)["boft"] == 1_410_048

    optimizer = build_optimizer(model, external_boft_lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), torch.tensor([2])).backward()
    diagnostics = layerwise_internal_external_diagnostics(model)
    assert diagnostics["left_rotation_gradient_norm"] > 0
    assert diagnostics["right_rotation_gradient_norm"] > 0
    assert diagnostics["external_left_rotation_gradient_norm"] > 0
    assert diagnostics["external_right_rotation_gradient_norm"] > 0
    external = model.external_boft
    assert external is not None
    assert torch.all(
        external.left_skew_parameters.grad.square().sum(dim=(1, 2, 3)) > 0
    )
    assert torch.all(
        external.right_skew_parameters.grad.square().sum(dim=(1, 2, 3)) > 0
    )
    print(
        "Layer-wise internal-external diagnostics: "
        f"initialization_error={(actual - expected).abs().max().item():.3e}, "
        f"external_left_gradient="
        f"{diagnostics['external_left_rotation_gradient_norm']:.3e}, "
        f"external_right_gradient="
        f"{diagnostics['external_right_rotation_gradient_norm']:.3e}"
    )


if __name__ == "__main__":
    test_layerwise_external_identity_and_independence()
    test_pretrained_layerwise_internal_external_model()
    print("Layer-wise internal-external QK-BOFT tests passed.")

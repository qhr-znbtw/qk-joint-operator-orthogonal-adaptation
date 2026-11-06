"""Diagnostics for internal-external bilateral QK-BOFT."""

from __future__ import annotations

from torch import nn

from experiments.qk_boft_common import (
    _parameter_norm,
    _rotation_distances,
    bilateral_diagnostics,
)


def internal_external_diagnostics(model: nn.Module) -> dict[str, float]:
    output = bilateral_diagnostics(model)
    external = model.external_boft
    if external is None:
        raise RuntimeError("External BOFT is not initialized.")

    for side in ("left", "right"):
        parameters = [getattr(external, f"{side}_skew_parameters")]
        blocks = getattr(external, f"{side}_orthogonal_blocks")().float()
        dense = getattr(external, f"dense_{side}_row_rotation")().float()
        block_mean, block_max = _rotation_distances(blocks)
        dense_mean, dense_max = _rotation_distances(dense.unsqueeze(0))
        output.update(
            {
                f"external_{side}_rotation_gradient_norm": _parameter_norm(
                    parameters, gradient=True
                ),
                f"external_{side}_rotation_parameter_norm": _parameter_norm(
                    parameters, gradient=False
                ),
                f"external_{side}_mean_block_rotation_distance": block_mean,
                f"external_{side}_max_block_rotation_distance": block_max,
                f"external_{side}_rotation_distance": dense_mean,
                f"external_{side}_max_rotation_distance": dense_max,
            }
        )
    return output

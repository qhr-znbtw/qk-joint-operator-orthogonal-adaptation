"""Train bilateral joint-QK BOFT with unconstrained positive spectrum."""

import torch

from experiments.qk_boft_bilateral_spectrum.model import (
    ViTQKBOFTBilateralSpectrumClassifier,
)
from experiments.qk_boft_common import (
    bilateral_diagnostics,
    parse_args,
    run_experiment,
)


@torch.no_grad()
def spectrum_diagnostics(model) -> dict[str, float]:
    output = bilateral_diagnostics(model)
    layers = model.adapter_layers()
    parameters = [layer.spectrum_log_scales for layer in layers]
    parameter_norm = torch.stack(
        [parameter.float().square().sum() for parameter in parameters]
    ).sum().sqrt()
    gradient_terms = [
        parameter.grad.float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    gradient_norm = (
        torch.stack(gradient_terms).sum().sqrt()
        if gradient_terms
        else parameter_norm.new_zeros(())
    )
    base = torch.cat([layer.base_singular_values() for layer in layers])
    learned = torch.cat([layer.learned_singular_values() for layer in layers])
    relative_change = torch.linalg.vector_norm(
        learned - base, dim=-1
    ) / torch.linalg.vector_norm(base, dim=-1).clamp_min(1e-12)
    ratios = learned / base.clamp_min(1e-12)
    output.update(
        {
            "spectrum_gradient_norm": float(gradient_norm.item()),
            "spectrum_parameter_norm": float(parameter_norm.item()),
            "mean_relative_spectrum_change": float(relative_change.mean().item()),
            "max_relative_spectrum_change": float(relative_change.max().item()),
            "min_singular_value_ratio": float(ratios.min().item()),
            "max_singular_value_ratio": float(ratios.max().item()),
        }
    )
    return output


def main() -> None:
    args = parse_args(__doc__ or "QK-BOFT-Bilateral-Learnable-Spectrum")
    run_experiment(
        args=args,
        model_factory=ViTQKBOFTBilateralSpectrumClassifier,
        method="qk_boft_bilateral_unconstrained_spectrum",
        label="QK-BOFT-Bilateral-Unconstrained-Spectrum",
        side="bilateral",
        expected_boft=138_240,
        expected_spectrum=9_216,
        spectrum_lr=args.spectrum_lr,
        spectrum_parameterization="unconstrained_log_scale",
        diagnostics_fn=spectrum_diagnostics,
    )


if __name__ == "__main__":
    main()

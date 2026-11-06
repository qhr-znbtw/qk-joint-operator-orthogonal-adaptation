"""Train full-spectrum bilateral joint-QK BOFT on Mini-VTAB-3."""

from experiments.qk_boft_bilateral.model import ViTQKBOFTBilateralClassifier
from experiments.qk_boft_common import (
    bilateral_diagnostics,
    parse_args,
    run_experiment,
)


def main() -> None:
    args = parse_args(__doc__ or "QK-BOFT-Full-Bilateral")
    run_experiment(
        args=args,
        model_factory=ViTQKBOFTBilateralClassifier,
        method="qk_boft_bilateral_full",
        label="QK-BOFT-Full-Bilateral",
        side="bilateral",
        expected_boft=138_240,
        diagnostics_fn=bilateral_diagnostics,
    )


if __name__ == "__main__":
    main()

"""Train full-spectrum right-sided joint-QK BOFT on Mini-VTAB-3."""

from experiments.qk_boft_common import (
    parse_args,
    run_experiment,
    single_side_diagnostics,
)
from experiments.qk_boft_right.model import ViTQKBOFTRightClassifier


def main() -> None:
    args = parse_args(__doc__ or "QK-BOFT-Full-Right")
    run_experiment(
        args=args,
        model_factory=ViTQKBOFTRightClassifier,
        method="qk_boft_right_full",
        label="QK-BOFT-Full-Right",
        side="right",
        expected_boft=69_120,
        diagnostics_fn=single_side_diagnostics,
    )


if __name__ == "__main__":
    main()

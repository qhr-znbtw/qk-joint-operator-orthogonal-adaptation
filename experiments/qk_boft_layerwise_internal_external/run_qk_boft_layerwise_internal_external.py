"""Train layer-wise external plus head-wise internal QK-BOFT on Mini-VTAB-3."""

from experiments.qk_boft_common import parse_args, run_experiment
from experiments.qk_boft_layerwise_internal_external.config import (
    EXTERNAL_BOFT_BLOCK_SIZE,
    EXTERNAL_BOFT_NUM_FACTORS,
    EXTERNAL_BOFT_PARAMETERS,
    INTERNAL_BOFT_PARAMETERS,
    TOTAL_BOFT_PARAMETERS,
)
from experiments.qk_boft_layerwise_internal_external.diagnostics import (
    layerwise_internal_external_diagnostics,
)
from experiments.qk_boft_layerwise_internal_external.model import (
    ViTQKBOFTLayerwiseInternalExternalClassifier,
)


def main() -> None:
    args = parse_args(__doc__ or "QK-BOFT-Layerwise-Internal-External")
    run_experiment(
        args=args,
        model_factory=ViTQKBOFTLayerwiseInternalExternalClassifier,
        method="qk_boft_layerwise_internal_external_bilateral",
        label="QK-BOFT-Layerwise-Internal-External-Bilateral",
        side="bilateral_layerwise_internal_external",
        expected_boft=TOTAL_BOFT_PARAMETERS,
        diagnostics_fn=layerwise_internal_external_diagnostics,
        external_boft_lr=args.external_boft_lr,
        extra_result_metadata={
            "external_sharing": "within_each_layer_across_12_heads",
            "external_bias_coordinate": "fixed",
            "external_boft_block_size": EXTERNAL_BOFT_BLOCK_SIZE,
            "external_boft_num_factors": EXTERNAL_BOFT_NUM_FACTORS,
            "external_boft_parameters": EXTERNAL_BOFT_PARAMETERS,
            "internal_boft_parameters": INTERNAL_BOFT_PARAMETERS,
        },
    )


if __name__ == "__main__":
    main()

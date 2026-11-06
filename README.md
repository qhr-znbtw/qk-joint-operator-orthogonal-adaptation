# Joint-QK Orthogonal Adaptation for Vision Transformers

This repository studies parameter-efficient adaptation of the **joint attention
operator** in a pretrained ViT-B/16. Instead of adapting `W_Q` and `W_K`
independently, each attention head is represented by

```text
M_h = W_{Q,h} W_{K,h}^T,
```

and a compact SVD is used to apply learnable rotations to its left singular
vectors, right singular vectors, or both. The study includes dense Cayley
rotations, BOFT-structured rotations, PSOFT-style support selection,
LOFT-style task-aware support, learnable singular spectra, and internal plus
external subspace rotations.

The main value of the repository is an extensive controlled result set,
including a clear negative result: under the current Mini-VTAB-3 protocol,
joint-QK adaptation does not outperform standard adaptation of individual
ViT weight matrices.

## Experimental Setting

- Backbone: `google/vit-base-patch16-224-in21k`
- Tasks: DTD, EuroSAT, and dSprites Orientation
- Development split: 800 training and 200 validation examples per task
- Evaluation: full test set
- Seed: 42
- Joint-QK methods: 100 epochs unless documented otherwise
- Data, pretrained weights, and checkpoints are excluded from Git

## Main Results

| Method | Adapter parameters | DTD | EuroSAT | dSprites | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard LoRA | 294,912 | 69.68 | 94.24 | 49.58 | 71.17 |
| Standard BOFT | 1,271,808 | 70.32 | 95.41 | 49.99 | **71.90** |
| QK-PSOFT bilateral | 580,608 | 68.35 | 93.76 | 46.17 | 69.43 |
| QK-BOFT bilateral | 138,240 | **69.89** | 92.91 | 47.11 | **69.97** |
| QK-BOFT + unconstrained spectrum | 147,456 | 69.15 | 92.94 | 46.92 | 69.67 |
| QK-BOFT + global external rotations | 244,224 | 68.88 | 93.22 | 46.74 | 69.61 |
| QK-BOFT + layer-wise external rotations | 1,410,048 | 68.51 | **93.74** | **47.14** | 69.80 |
| Best QK-LOFT configuration | 149,040 | 68.88 | 93.11 | 46.84 | 69.61 |

Values are percentages. These are single-seed results; differences below
approximately one percentage point should not be interpreted as statistically
conclusive.

The layer-wise internal/external design uses more parameters than standard
BOFT while scoring 2.10 points lower in mean accuracy. This suggests that the
limitation is structural rather than a missing local rotation parameterization:
forming `W_Q W_K^T` removes independent `Q`/`K` degrees of freedom, while
internal SVD rotations preserve fixed ambient subspaces and external rotations
recover that mobility only at substantial parameter cost.

See [the complete experiment summary](reports/all_experiment_results_summary.md)
for every result and protocol caveat.

## Implemented Method Families

```text
experiments/
├── linear_probe/                         # frozen-backbone baseline
├── lora/                                 # standard LoRA baseline
├── oft/                                  # standard block-OFT baseline
├── qk_svd_oft/                           # common joint-QK/SVD machinery
├── qk_psoft_left/                        # dense left rotation
├── qk_psoft_right/                       # dense right rotation
├── qk_psoft_bilateral/                   # dense bilateral rotation
├── qk_boft_left/                         # BOFT left rotation
├── qk_boft_right/                        # BOFT right rotation
├── qk_boft_bilateral/                    # BOFT bilateral rotation
├── qk_boft_bilateral_spectrum/           # learnable singular spectrum
├── qk_loft/                              # Principal/GradSVD/SkewGrad support
├── qk_boft_internal_external/            # globally shared external rotations
└── qk_boft_layerwise_internal_external/  # layer-wise external rotations
```

Shared ViT, data-loading, training, and evaluation code lives in `vit_core/`.
Formal JSON outputs are in `results/`, detailed interpretations are in
`reports/`, and algebra/integration checks are in `tests/`.

## Installation and Preparation

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
python -m scripts.download_mini_vtab
python -m scripts.prepare_dsprites_orientation
python -m scripts.download_pretrained_model
```

## Running Representative Experiments

Each command accepts `dtd`, `eurosat`, or `dsprites_orientation`.

```bash
# Dense joint-QK rotations
python -m experiments.qk_psoft_left.run_qk_psoft_left --task dtd
python -m experiments.qk_psoft_right.run_qk_psoft_right --task dtd
python -m experiments.qk_psoft_bilateral.run_qk_psoft_bilateral --task dtd

# Structured BOFT rotations
python -m experiments.qk_boft_left.run_qk_boft_left --task dtd
python -m experiments.qk_boft_right.run_qk_boft_right --task dtd
python -m experiments.qk_boft_bilateral.run_qk_boft_bilateral --task dtd
```

The exact entry points and hyperparameters for the larger factorial studies are
documented in their corresponding experiment directories and reports.

## Verification

Run the complete lightweight test suite with:

```bash
python -m pytest tests
```

The tests cover head decomposition, exact initialization, pretrained loading,
gradient flow, orthogonality, spectrum parameterization, and each joint-QK
adapter family.

## Interpretation and Scope

This is an exploratory undergraduate research codebase, not a claim of a new
state-of-the-art method. It is retained because it provides:

1. a transparent implementation of SVD-based joint-attention adaptation;
2. matched left/right/bilateral and support-selection ablations;
3. parameter, timing, memory, and accuracy evidence for unsuccessful variants;
4. a reproducible basis for analyzing why adapting a composed operator can be
   weaker than adapting its factors independently.

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Orthogonal Finetuning](https://arxiv.org/abs/2306.07280)

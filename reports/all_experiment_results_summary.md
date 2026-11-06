# Complete Mini-VTAB-3 Experiment Summary

This report consolidates the non-duplicated formal results stored across the
baseline, OFT, HRA/HOFT, PSOFT, LOFT, and joint-QK projects. Pilot 20-epoch
linear-probe runs are excluded.

All runs use the same ViT-B/16 pretrained backbone, fixed 800/200 task split,
full test sets, and seed 42. However, the training recipes are not identical:

- the baseline/OFT/HRA/HOFT family uses 100 epochs;
- PSOFT and standard LOFT use the paper-oriented 50-epoch recipe and
  task-specific adapter learning rates;
- joint-QK experiments use 100 epochs.

The tables therefore support controlled comparisons within each protocol more
strongly than a universal ranking across every row.

## 1. Adaptation of individual ViT weight matrices

| Method | Epochs | Adapter params | DTD | EuroSAT | dSprites | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 100 | 0 | 67.13% | 89.24% | 24.48% | 60.28% |
| LoRA | 100 | 294,912 | 69.68% | 94.24% | 49.58% | 71.17% |
| Block OFT | 100 | 211,968 | 70.27% | 93.76% | 46.07% | 70.03% |
| BOFT | 100 | 1,271,808 | 70.32% | 95.41% | 49.99% | 71.90% |
| GOFT | 100 | 18,408 | 69.52% | 93.52% | 45.41% | 69.48% |
| HRA | 100 | 147,456 | 68.67% | 91.80% | 38.16% | 66.21% |
| HOFT | 100 | 147,456 | 69.26% | 94.87% | 46.86% | 70.33% |
| PSOFT strict | 50 | 74,520 | 69.89% | 94.57% | 47.69% | 70.72% |
| PSOFT scaled | 50 | 81,144 | 69.68% | 94.56% | 49.04% | 71.09% |
| LOFT Principal | 50 | 74,520 | 69.79% | **96.33%** | 45.31% | 70.48% |
| LOFT GradSVD | 50 | 74,520 | 70.05% | 95.85% | 49.72% | 71.88% |
| LOFT SkewGrad | 50 | 74,520 | **70.59%** | 94.19% | **52.24%** | **72.34%** |

The strongest 100-epoch common-protocol method is BOFT at 71.90% mean, but it
uses 1.27 million adapter parameters. Under the 50-epoch LOFT/PSOFT protocol,
LOFT SkewGrad gives the strongest mean at 72.34% with 74,520 adapter
parameters. Because the targets, learning rates, dropout, and epoch budgets
differ, this is not a strictly matched BOFT-versus-LOFT comparison.

## 2. Adaptation of the per-head joint QK operator

### QK-PSOFT and QK-BOFT

| Method | Adapter params | DTD | EuroSAT | dSprites | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| QK-PSOFT left | 290,304 | 68.46% | 93.07% | 46.88% | 69.47% |
| QK-PSOFT right | 290,304 | 68.99% | 93.04% | 45.68% | 69.24% |
| QK-PSOFT bilateral | 580,608 | 68.35% | 93.76% | 46.17% | 69.43% |
| QK-BOFT left | 69,120 | 69.52% | 91.98% | 44.66% | 68.72% |
| QK-BOFT right | 69,120 | 66.97% | 92.28% | 45.42% | 68.22% |
| QK-BOFT bilateral | 138,240 | **69.89%** | 92.91% | 47.11% | **69.97%** |
| QK-BOFT internal + global external | 244,224 | 68.88% | 93.22% | 46.74% | 69.61% |
| QK-BOFT internal + layer-wise external | 1,410,048 | 68.51% | **93.74%** | **47.14%** | 69.80% |
| QK-BOFT bilateral + norm-preserving spectrum | 147,456 | 68.35% | 93.07% | 47.05% | 69.49% |
| QK-BOFT bilateral + unconstrained spectrum | 147,456 | 69.15% | 92.94% | 46.92% | 69.67% |

QK-BOFT bilateral with a fixed spectrum is the strongest non-LOFT joint-QK
configuration.  Removing centering and L2-norm restoration from the learnable
spectrum raises its mean from 69.49% to 69.67%, but still trails the fixed
spectrum by 0.30 points.  Thus neither spectrum parameterization improves the
current single-seed aggregate result.  Adding two globally shared ambient
BOFT rotations moves the left and right singular subspaces and improves
EuroSAT by 0.31 points, but reduces DTD and dSprites by 1.01 and 0.37 points;
its 69.61% mean is 0.36 points below internal bilateral BOFT.  Replacing the
global pair with an independent pair per layer raises EuroSAT and dSprites to
93.74% and 47.14%, but lowers DTD to 68.51%.  Its 69.80% mean recovers 0.19
points over global sharing but still trails internal-only BOFT by 0.17 points
despite using 10.2 times as many adapter parameters.

### QK-LOFT: support by side

| Support | Side | Adapter params | DTD | EuroSAT | dSprites | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Principal | Left | 149,040 | **68.88%** | 93.11% | **46.84%** | **69.61%** |
| Principal | Right | 149,040 | 68.83% | 93.00% | 45.88% | 69.24% |
| Principal | Bilateral | 298,080 | 68.56% | 94.15% | 45.54% | 69.42% |
| GradSVD | Left | 149,040 | 68.46% | 93.09% | 45.56% | 69.04% |
| GradSVD | Right | 149,040 | 67.98% | 92.96% | 45.20% | 68.71% |
| GradSVD | Bilateral | 298,080 | 67.55% | 93.81% | 46.80% | 69.39% |
| SkewGrad | Left | 149,040 | 68.67% | 93.41% | 46.14% | 69.41% |
| SkewGrad | Right | 149,040 | 68.30% | 93.17% | 44.62% | 68.69% |
| SkewGrad | Bilateral | 298,080 | 67.02% | **94.20%** | 46.29% | 69.17% |

Principal-left is the strongest QK-LOFT configuration at 69.61% mean. Averaged
over side choices, Principal support scores 69.42%, SkewGrad 69.09%, and
GradSVD 69.05%. In contrast to standard LOFT, task-aware support does not help
the current joint-QK formulation.

## 3. Main findings

1. Standard PEFT remains stronger than the current joint-QK methods. LoRA,
   standard BOFT, and standard LOFT all exceed the best joint-QK mean.
2. Within joint-QK adaptation, fixed-spectrum QK-BOFT bilateral is best at
   69.97%; layer-wise internal-external BOFT reaches 69.80%, and its
   unconstrained-spectrum variant reaches 69.67%.  Moving the singular
   subspaces therefore helps selected tasks but has not improved the aggregate.
3. Left adaptation is generally more parameter-efficient than right or
   bilateral QK-LOFT. Bilateral capacity mainly helps EuroSAT and does not
   reliably improve the mean.
4. The contrast between standard LOFT and QK-LOFT is important: gradient-aware
   support substantially helps standard weights but not the fixed compact QK
   subspaces.
5. A plausible structural explanation is that
   $M'=US_L\Sigma S_R^TV^T$ preserves both the left/right ambient subspaces and
   the singular spectrum. The next experiment should therefore test a
   low-parameter external-subspace rotation rather than another internal
   support parameterization.

## 4. Evidence limits

- Every result is single-seed (42).
- Differences below roughly one percentage point require multi-seed
  confirmation.
- Standard LOFT calibration in the local reproduction uses one mini-batch,
  whereas the paper's default GradSVD construction accumulates four batches.
- Some method families adapt different sets of ViT linear layers and use
  different epoch or learning-rate protocols.

# Joint-QK LOFT: Support-by-Side Factorial Study

## Research question

This experiment asks whether LOFT-style fixed low-dimensional support selection
remains useful when adaptation is applied to the **joint attention operator**
rather than to an individual pretrained weight. For attention head $h$, define

$$
M_h = \widetilde W_{Q,h}\widetilde W_{K,h}^{\top}
    = U_h\Sigma_hV_h^{\top},
$$

where the tilde denotes the bias-augmented factors and
$U_h,V_h\in\mathbb R^{769\times64}$. We compare three fixed rank-46 supports
and three placements of the learned orthogonal transform.

## Adapter

For a row-orthonormal support $P\in\mathbb R^{46\times64}$ and a learned
$T\in O(46)$, the embedded transform is

$$
S = I_{64}+P^{\top}(T-I_{46})P.
$$

$T$ is parameterized by an exact Cayley map initialized at identity:

$$
T=(I+A)^{-1}(I-A),\qquad A^{\top}=-A.
$$

The three placements are

$$
\begin{aligned}
\text{left:}\quad &M_h'=U_hS_{L,h}\Sigma_hV_h^{\top},\\
\text{right:}\quad &M_h'=U_h\Sigma_hS_{R,h}^{\top}V_h^{\top},\\
\text{bilateral:}\quad
&M_h'=U_hS_{L,h}\Sigma_hS_{R,h}^{\top}V_h^{\top}.
\end{aligned}
$$

The implementation refactorizes each adapted operator into compatible Query
and Key factors, so regular scaled dot-product attention is retained without
materializing a $769\times769$ matrix.

## Fixed supports

- **Principal** selects the first 46 compact singular coordinates.
- **GradSVD** uses one training-only calibration batch. If
  $G_h=\partial\mathcal L/\partial B_h$ at $B_h=\Sigma_h$, the left and right
  supports are the leading left and right singular spaces of $G_h$.
- **SkewGrad** uses the dominant paired subspaces of
  $\operatorname{skew}(G_h\Sigma_h^{\top})$ on the left and
  $\operatorname{skew}(\Sigma_h^{\top}G_h)$ on the right.

Support matrices are fixed buffers. Only the independent upper-triangular
coordinates of the rank-46 skew matrices are optimized.

## Controlled protocol

| Setting | Value |
| --- | --- |
| Backbone | ImageNet-21k pretrained ViT-B/16 |
| Adapted operators | 12 layers x 12 heads = 144 joint QK operators |
| Support rank | 46 of 64 compact dimensions |
| Tasks | DTD, EuroSAT, dSprites Orientation |
| Split | Fixed 800 train / 200 validation / full test |
| Epochs | 100 |
| Warmup and schedule | 10 epochs, then cosine decay |
| Adapter learning rate | `5e-4` |
| Classifier learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Calibration | One training batch for GradSVD and SkewGrad; none for Principal |
| Random seed | 42 |
| Precision | BF16 automatic mixed precision |

The calibration loader is independent of the formal training loader. Formal
loaders are rebuilt after support construction, so calibration does not change
the fixed shuffle order. Validation and test examples are never used to select
supports.

## Complete 3 x 3 x 3 results

Test accuracy uses the best validation checkpoint from each 100-epoch run.

| Support | Side | DTD | EuroSAT | dSprites | Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Principal | Left | **68.88%** | 93.11% | **46.84%** | **69.61%** |
| Principal | Right | 68.83% | 93.00% | 45.88% | 69.24% |
| Principal | Bilateral | 68.56% | 94.15% | 45.54% | 69.42% |
| GradSVD | Left | 68.46% | 93.09% | 45.56% | 69.04% |
| GradSVD | Right | 67.98% | 92.96% | 45.20% | 68.71% |
| GradSVD | Bilateral | 67.55% | 93.81% | 46.80% | 69.39% |
| SkewGrad | Left | 68.67% | 93.41% | 46.14% | 69.41% |
| SkewGrad | Right | 68.30% | 93.17% | 44.62% | 68.69% |
| SkewGrad | Bilateral | 67.02% | **94.20%** | 46.29% | 69.17% |

The best mean is **Principal-left at 69.61%**. Task-aware supports do not
improve the mean under this one-batch calibration protocol. Bilateral
adaptation gives the strongest EuroSAT results but usually loses accuracy on
DTD, so doubling the number of rotation parameters is not uniformly useful.

## Marginal comparison

Average over the other factorial axis:

| Factor | Level | Mean accuracy |
| --- | --- | ---: |
| Support | Principal | **69.42%** |
| Support | GradSVD | 69.05% |
| Support | SkewGrad | 69.09% |
| Side | Left | **69.35%** |
| Side | Right | 68.88% |
| Side | Bilateral | 69.33% |

The left side is the most parameter-efficient placement: its marginal mean is
slightly higher than bilateral while using half as many adapter parameters.
The right side is consistently weakest, particularly on dSprites.

## Parameters, runtime, and memory

Each active head-side contains

$$
\frac{46\cdot45}{2}=1{,}035
$$

trainable skew coordinates. Therefore:

| Side | Adapter parameters | Peak training VRAM |
| --- | ---: | ---: |
| Left or right | 149,040 | 5.33 GiB |
| Bilateral | 298,080 | 5.58 GiB |

Across the 27 corrected formal runs, total training time was approximately
8,879 seconds (2.47 hours), excluding test evaluation and the period for which
the computer was suspended. One-batch task-aware support construction took
about 0.9--1.7 seconds per run. Principal construction was effectively
instantaneous.

## Comparison with existing joint-QK methods

| Method | DTD | EuroSAT | dSprites | Mean | Adapter parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| QK-PSOFT left, rank 64 | 68.46% | 93.07% | 46.88% | 69.47% | 290,304 |
| QK-PSOFT bilateral, rank 64 | 68.35% | 93.76% | 46.17% | 69.43% | 580,608 |
| QK-BOFT bilateral | **69.89%** | 92.91% | **47.11%** | **69.97%** | 138,240 |
| QK-LOFT Principal-left, rank 46 | 68.88% | 93.11% | 46.84% | 69.61% | 149,040 |
| QK-LOFT SkewGrad-bilateral, rank 46 | 67.02% | **94.20%** | 46.29% | 69.17% | 298,080 |

QK-LOFT Principal-left improves the mean over dense rank-64 QK-PSOFT while
using 48.7% fewer adapter parameters. It does not exceed QK-BOFT bilateral,
which remains both more accurate on average and smaller. The result suggests
that support restriction is viable, but the current one-batch task-aware
selection is not yet the main source of improvement.

## Interpretation and next experiments

1. **Principal support is a strong default.** The compact QK singular basis
   already concentrates useful adaptation directions; a noisy single-batch
   gradient can select a worse rank-46 subspace.
2. **Left and right are not empirically symmetric.** Although both are valid
   orthogonal changes of compact coordinates, the pretrained $U$ and $V$
   spaces and downstream gradients play different roles.
3. **Bilateral capacity is task dependent.** It improves EuroSAT but often
   hurts DTD. A shared budget or coupling penalty is more justified than simply
   assigning two independent rotations.
4. **The next high-value ablation is calibration quality.** Test
   $K\in\{1,2,4,8\}$ calibration batches for SkewGrad-left before changing the
   rank. If the result remains flat, support learning or a soft mixture between
   Principal and task-aware support is more promising.
5. **Multi-seed confirmation is required.** All current results use seed 42;
   differences below roughly one percentage point should not be presented as
   statistically conclusive.

## Reproduction

Run one configuration:

```bash
python -m experiments.qk_loft.run_qk_loft \
  --support principal --side left --task dtd
```

Run or resume all 27 experiments:

```bash
python -m scripts.run_qk_loft_factorial
```

Machine-readable outputs are stored under
`results/qk_loft_<support>_<side>/development_800_200/<task>/seed_42.json`.
Per-run console logs are stored under `results/logs/qk_loft_factorial/`.

## Verification performed

The implementation tests cover:

- exact equality between the original attention and the compact calibration
  operator at initialization;
- row orthonormality for Principal, GradSVD, and SkewGrad supports;
- equality between the implicit support update and its dense embedded formula;
- equality between direct joint-QK logits and logits from merged Query/Key
  factors for left, right, and bilateral variants;
- nonzero gradients for every active side;
- exact adapter counts of 149,040 and 298,080.

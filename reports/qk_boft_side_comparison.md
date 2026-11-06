# Joint-QK BOFT: Left, Right, and Bilateral Comparison

## Methods

For every bias-aware ViT attention-head operator

\[
M_h=\widetilde W_{Q,h}\widetilde W_{K,h}^{\mathsf T}
   =U_h\Sigma_hV_h^{\mathsf T},
\]

we compare three full-spectrum BOFT adaptations:

\[
\begin{aligned}
M_h^{(L)} &= U_hR_{L,h}^{\mathrm{BOFT}}\Sigma_hV_h^{\mathsf T},\\
M_h^{(R)} &= U_h\Sigma_h(R_{R,h}^{\mathrm{BOFT}})^{\mathsf T}V_h^{\mathsf T},\\
M_h^{(LR)} &= U_hR_{L,h}^{\mathrm{BOFT}}\Sigma_h
              (R_{R,h}^{\mathrm{BOFT}})^{\mathsf T}V_h^{\mathsf T}.
\end{aligned}
\]

Each BOFT rotation uses block size 4 and five independent butterfly factors.
The factors use fixed permutations with micro-block strides \(1,2,4,8,16\),
giving full 64-dimensional information coverage.  Every \(4\times4\) block is
an identity-initialized scaled-Cayley rotation.  The left and right rotations
in the bilateral model have separate learned parameters.

All methods retain all 64 singular directions and keep the singular values
fixed.  They use the same Mini-VTAB-3 splits, seed 42, optimizer, learning
rates, 10-epoch warmup, cosine schedule, batch sizes, and 100-epoch budget.

## Test accuracy

| Method | BOFT parameters | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| QK-BOFT Full-Left | 69,120 | 69.52 | 91.98 | 44.66 | 68.72 |
| QK-BOFT Full-Right | 69,120 | 66.97 | 92.28 | 45.42 | 68.22 |
| **QK-BOFT Full-Bilateral** | **138,240** | **69.89** | **92.91** | **47.11** | **69.97** |

The bilateral variant is the strongest BOFT version on every task.  Relative
to left-only BOFT, it improves DTD by 0.37 points, EuroSAT by 0.93 points, and
dSprites Orientation by 2.45 points, for a mean improvement of 1.25 points.
Relative to right-only BOFT, the mean improvement is 1.75 points.

The single-side comparison is task-dependent.  Left rotation is much stronger
on DTD (+2.55 points), whereas right rotation is moderately stronger on
EuroSAT (+0.30) and dSprites (+0.76).  Consequently, neither singular
subspace is universally preferable.

## Run details

| Method | Task | Best epoch | Best validation | Test accuracy | Training time | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Left | DTD | 29 | 67.50 | 69.52 | 344.30 s | 5.39 GiB |
| Left | EuroSAT | 33 | 90.50 | 91.98 | 297.59 s | 5.39 GiB |
| Left | dSprites | 39 | 44.50 | 44.66 | 292.41 s | 5.39 GiB |
| Right | DTD | 17 | 66.50 | 66.97 | 329.44 s | 5.39 GiB |
| Right | EuroSAT | 57 | 91.00 | 92.28 | 296.00 s | 5.39 GiB |
| Right | dSprites | 44 | 44.00 | 45.42 | 292.88 s | 5.39 GiB |
| Bilateral | DTD | 44 | 66.50 | 69.89 | 348.64 s | 5.70 GiB |
| Bilateral | EuroSAT | 44 | 92.00 | 92.91 | 316.66 s | 5.70 GiB |
| Bilateral | dSprites | 41 | 47.00 | 47.11 | 313.34 s | 5.70 GiB |

Total training times are 934.30 seconds for left, 918.32 seconds for right,
and 978.64 seconds for bilateral BOFT.  Doubling the BOFT parameters increases
peak memory by approximately 0.31 GiB and total training time by only about
5% relative to the left variant.

## Rotation diagnostics at the selected checkpoint

The normalized composed-rotation distance is

\[
d(R,I)=\frac{\lVert R-I\rVert_F}{\sqrt{64}}.
\]

| Method | Task | Left / single distance | Right distance |
| --- | --- | ---: | ---: |
| Left | DTD | 0.23 | -- |
| Left | EuroSAT | 0.25 | -- |
| Left | dSprites | 0.30 | -- |
| Right | DTD | 0.16 | -- |
| Right | EuroSAT | 0.29 | -- |
| Right | dSprites | 0.32 | -- |
| Bilateral | DTD | 0.22 | 0.22 |
| Bilateral | EuroSAT | 0.20 | 0.20 |
| Bilateral | dSprites | 0.23 | 0.23 |

Both bilateral sides receive nonzero gradients.  Their learned distances are
nearly equal under the symmetric identity initialization.  The gain therefore
does not come from one side remaining inactive; both sides learn comparable,
moderate rotations whose composition is more expressive than either alone.

## Comparison with dense Joint-QK rotations and standard baselines

| Method | Adapter / rotation parameters | DTD | EuroSAT | dSprites | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probe | 0 | 67.13 | 89.24 | 24.48 | 60.28 |
| LoRA | 294,912 | 69.68 | **94.24** | **49.58** | **71.17** |
| Block-OFT | 211,968 | **70.27** | 93.76 | 46.07 | 70.03 |
| Dense QK-PSOFT Full-Bilateral | 580,608 | 68.35 | 93.76 | 46.17 | 69.43 |
| **QK-BOFT Full-Bilateral** | **138,240** | 69.89 | 92.91 | 47.11 | 69.97 |

The structured bilateral model uses only 23.8% as many rotation parameters as
the dense bilateral model, yet improves its mean by 0.54 points.  It is also
within 0.06 points of the Block-OFT mean, although the task-level behavior is
different.  LoRA remains the strongest overall baseline.

## Conclusions

1. Right-sided and bilateral Joint-QK BOFT are mathematically and
   computationally valid: they reproduce the pretrained network at identity,
   preserve orthogonality, receive gradients, and can be merged into ordinary
   query/key parameters.
2. The side of spectral rotation matters.  Left is preferable on DTD, while
   right is slightly preferable on EuroSAT and dSprites.
3. Bilateral structured rotation is the strongest BOFT configuration and
   recovers much of the capacity lost by either single-side model.
4. BOFT is not merely a compressed approximation to dense rotation.  In the
   bilateral comparison, its structural constraint improves mean accuracy
   despite using 76.2% fewer rotation parameters.
5. These are single-seed results.  Repeated seeds are required before treating
   differences below roughly one percentage point as stable effects.

The next useful ablation is to separate mixing depth from block capacity:
compare \((b,L)=(4,5)\), \((4,6)\), and \((8,4)\) for the bilateral model while
holding the data split and optimization protocol fixed.

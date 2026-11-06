# Left- versus Right-Sided Joint-QK PSOFT

## Controlled question

For each ViT-B/16 attention head, let the frozen bias-aware joint operator be

\[
M_h=\widetilde W_{Q,h}\widetilde W_{K,h}^{\mathsf T}
   =U_h\Sigma_hV_h^{\mathsf T}.
\]

The two experiments differ in exactly one choice:

\[
\begin{aligned}
\text{Left:}\quad
M_h^{(L)} &= U_hR_h\Sigma_hV_h^{\mathsf T},\\
\text{Right:}\quad
M_h^{(R)} &= U_h\Sigma_hR_h^{\mathsf T}V_h^{\mathsf T}.
\end{aligned}
\]

Both methods adapt all 144 heads at compact rank 64 with identity-initialized
dense scaled-Cayley rotations.  They have the same 290,304 rotation
parameters, optimizer, learning rates, data splits, seed, and 100-epoch
training budget.

## Test accuracy

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13 | 89.24 | 24.48 | 60.28 |
| LoRA | 69.68 | **94.24** | **49.58** | **71.17** |
| Block-OFT | **70.27** | 93.76 | 46.07 | 70.03 |
| QK-PSOFT Full-Left | 68.46 | 93.07 | 46.88 | 69.47 |
| QK-PSOFT Full-Right | 68.99 | 93.04 | 45.68 | 69.24 |

Right minus left is +0.53 points on DTD, -0.03 on EuroSAT, and -1.20 on
dSprites Orientation.  The right-sided three-task mean is 0.23 points below
the left-sided mean.

## Right-sided run details

| Task | Best epoch | Best validation accuracy | Test accuracy | Training time | Peak VRAM | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 19 | 65.00 | 68.99 | 346.70 s | 5.34 GiB | 326,447 |
| EuroSAT | 39 | 93.50 | 93.04 | 299.69 s | 5.34 GiB | 297,994 |
| dSprites Orientation | 33 | 52.00 | 45.68 | 286.03 s | 5.34 GiB | 302,608 |

The three right-sided runs required 932.42 seconds in total.  The classifier
size changes with the number of task classes; the rotation component remains
fixed at 290,304 parameters.

## Rotation diagnostics

The reported rotation distance is

\[
d(R,I)=\frac{\lVert R-I\rVert_F}{\sqrt{64}}.
\]

| Task | Distance at best epoch (mean / max) | Distance at final epoch (mean / max) |
| --- | ---: | ---: |
| DTD | 0.180 / 0.220 | 0.270 / 0.320 |
| EuroSAT | 0.180 / 0.230 | 0.190 / 0.250 |
| dSprites Orientation | 0.210 / 0.270 | 0.230 / 0.310 |

All right-sided rotations moved away from identity and remained numerically
stable.  As with the left-sided experiment, DTD and dSprites reached their
best validation checkpoints before the end of training and subsequently
overfit.

## Interpretation

There is no evidence that one single-sided rotation is uniformly superior.
The two methods are effectively tied on EuroSAT, right rotation is slightly
better on DTD, and left rotation is better on dSprites.  The validation and
test rankings also disagree on DTD and dSprites: right rotation obtains a
lower DTD validation score but a higher DTD test score, while obtaining a
higher dSprites validation score but a lower dSprites test score.  The
200-image validation split is therefore too noisy to support a strong claim
about side preference from one seed.

The useful result is structural: left and right rotations are both trainable,
stable, and competitive with block-OFT, but they modify different singular
subspaces and produce measurably different task behavior.  This justifies the
next controlled experiment, a two-sided operator

\[
M_h^{(LR)}=U_hR_{L,h}\Sigma_hR_{R,h}^{\mathsf T}V_h^{\mathsf T},
\]

while also warning that its doubled rotation capacity may intensify
overfitting.  The first two-sided run should keep the same optimization
protocol so that the effect of added expressivity can be isolated.

All reported results use one split and seed 42; repeated seeds are required
before making a statistical claim.

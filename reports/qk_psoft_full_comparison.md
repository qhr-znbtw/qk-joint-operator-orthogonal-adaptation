# Full-Rank Joint-QK PSOFT: Left, Right, and Bilateral Rotations

## Experimental design

For each bias-aware ViT attention-head operator

\[
M_h=\widetilde W_{Q,h}\widetilde W_{K,h}^{\mathsf T}
   =U_h\Sigma_hV_h^{\mathsf T},
\]

the three proposed adaptations are

\[
\begin{aligned}
M_h^{(L)} &= U_hR_{L,h}\Sigma_hV_h^{\mathsf T},\\
M_h^{(R)} &= U_h\Sigma_hR_{R,h}^{\mathsf T}V_h^{\mathsf T},\\
M_h^{(LR)} &= U_hR_{L,h}\Sigma_hR_{R,h}^{\mathsf T}V_h^{\mathsf T}.
\end{aligned}
\]

Every rotation is a dense, identity-initialized, scaled-Cayley rotation in
dimension 64.  All 12 heads in all 12 Transformer blocks are adapted.  The
three experiments use the same Mini-VTAB-3 splits, seed 42, AdamW optimizer,
learning rates, weight decay, warmup, cosine schedule, batch sizes, and
100-epoch training budget.

## Test accuracy

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13 | 89.24 | 24.48 | 60.28 |
| LoRA | 69.68 | **94.24** | **49.58** | **71.17** |
| Block-OFT | **70.27** | 93.76 | 46.07 | 70.03 |
| QK-PSOFT Full-Left | 68.46 | 93.07 | 46.88 | 69.47 |
| QK-PSOFT Full-Right | 68.99 | 93.04 | 45.68 | 69.24 |
| QK-PSOFT Full-Bilateral | 68.35 | 93.76 | 46.17 | 69.43 |

The bilateral method improves over the stronger single-sided result by 0.69
points on EuroSAT, but is lower by 0.64 points on DTD and 0.71 points on
dSprites Orientation.  Its mean is 0.04 points below the left-sided method
and 0.19 points above the right-sided method.  Doubling the rotation capacity
therefore produces no aggregate improvement.

## Bilateral run details

| Task | Best epoch | Best validation accuracy | Test accuracy | Training time | Peak VRAM | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 30 | 64.50 | 68.35 | 355.18 s | 5.60 GiB | 616,751 |
| EuroSAT | 20 | 95.00 | 93.76 | 297.34 s | 5.60 GiB | 588,298 |
| dSprites Orientation | 19 | 51.00 | 46.17 | 300.15 s | 5.60 GiB | 592,912 |

The three bilateral runs required 952.67 seconds.  Each side contributes
290,304 rotation parameters, for 580,608 rotation parameters in total.  Peak
VRAM increased from approximately 5.34 GiB for a single side to 5.60 GiB.

## Independent side diagnostics

The normalized distance of a rotation from identity is

\[
d(R,I)=\frac{\lVert R-I\rVert_F}{\sqrt{64}}.
\]

| Task | Best epoch | Left mean distance | Right mean distance | Left gradient norm | Right gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| DTD | 30 | 0.1730 | 0.1730 | 0.0387 | 0.0387 |
| EuroSAT | 20 | 0.1130 | 0.1121 | 0.0256 | 0.0260 |
| dSprites Orientation | 19 | 0.1229 | 0.1236 | 0.8298 | 0.7635 |

Both sides receive nonzero gradients and move away from identity.  However,
their learned distances and gradient scales are strikingly similar.  In this
identity-initialized, symmetric setup, the optimizer does not appear to use
the two sides as strongly differentiated mechanisms.  This is empirical
evidence, not a proof of parameter redundancy, but it helps explain why twice
the rotation count does not yield twice the useful adaptation.

## Main findings

1. Joint-QK spectral rotation is a valid and trainable adaptation mechanism.
   All three variants substantially outperform Linear Probe.
2. There is no universally best rotation side.  Task behavior differs, but
   left and right have similar aggregate performance.
3. Bilateral rotation helps EuroSAT, yet provides no mean gain and overfits
   DTD and dSprites more quickly.  Extra expressivity alone is insufficient.
4. The best bilateral checkpoints occur at epochs 30, 20, and 19, earlier
   than most single-sided checkpoints.  This supports rank control,
   regularization, or early stopping as the next research variables.
5. LoRA remains the strongest mean baseline and Block-OFT remains stronger on
   DTD, so the current result establishes feasibility and a meaningful
   ablation rather than state-of-the-art performance.

Because all results use one split and one seed, repeated seeds are required
before making a statistical claim.  The most informative next experiment is
not another unrestricted full-rank variant, but a controlled reduction of
bilateral capacity, such as lower spectral rank or a coupling constraint
between the two rotations.

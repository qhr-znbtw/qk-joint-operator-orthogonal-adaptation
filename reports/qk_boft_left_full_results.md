# Full-Spectrum Joint-QK BOFT Results

## Research question

The previous full-rank Joint-QK experiment learned one unrestricted dense
rotation for every attention head.  This experiment asks whether a structured
butterfly product can retain useful adaptation while using substantially fewer
parameters.

For each bias-aware attention-head operator,

\[
M_h=\widetilde W_{Q,h}\widetilde W_{K,h}^{\mathsf T}
   =U_h\Sigma_hV_h^{\mathsf T},
\]

the new method learns

\[
\boxed{
M_h'=U_hR_h^{\mathrm{BOFT}}\Sigma_hV_h^{\mathsf T}
}
\]

over all 64 compact singular directions.  Thus, this experiment changes the
parameterization of the left rotation but does not discard any singular
component.

## BOFT parameterization

Each head has dimension 64.  A BOFT factor contains sixteen independent
\(4\times4\) rotations.  Every block is identity-initialized and is mapped
from six skew-symmetric coordinates through the scaled Cayley transform

\[
R(A)=\left(I+\tfrac12 A\right)^{-1}
     \left(I-\tfrac12 A\right),
\qquad A^{\mathsf T}=-A.
\]

Five factors use fixed binary-butterfly permutations with micro-block strides
\(1,2,4,8,16\).  Their effective information coverage therefore grows as

\[
4\longrightarrow8\longrightarrow16\longrightarrow32\longrightarrow64.
\]

The number of learned rotation parameters is

\[
144\text{ heads}\times5\text{ factors}\times16\text{ blocks}
\times\frac{4(4-1)}2=69{,}120.
\]

The dense single-sided rotation uses 290,304 parameters.  BOFT therefore uses
23.8% as many rotation parameters, a reduction of 76.2%.

## Experimental protocol

The experiment uses the same Mini-VTAB-3 development protocol as all earlier
runs: 800 training examples, 200 validation examples, seed 42, 100 epochs,
10 warmup epochs, AdamW, cosine decay, batch sizes 64/128, and selection of the
checkpoint with the highest validation accuracy.  The BOFT and classifier
learning rates are \(5\times10^{-4}\) and \(10^{-3}\), respectively, with
weight decay \(10^{-4}\).

## QK-BOFT run details

| Task | Best epoch | Best validation | Test accuracy | Training time | Peak VRAM | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 29 | 67.50 | 69.52 | 344.30 s | 5.39 GiB | 105,263 |
| EuroSAT | 33 | 90.50 | 91.98 | 297.59 s | 5.39 GiB | 76,810 |
| dSprites Orientation | 39 | 44.50 | 44.66 | 292.41 s | 5.39 GiB | 81,424 |

The three training runs required 934.30 seconds in total.

## Comparison with previous methods

| Method | Rotation / adapter parameters | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probe | 0 | 67.13 | 89.24 | 24.48 | 60.28 |
| LoRA | 294,912 | 69.68 | **94.24** | **49.58** | **71.17** |
| Block-OFT | 211,968 | **70.27** | 93.76 | 46.07 | 70.03 |
| QK-PSOFT Full-Left | 290,304 | 68.46 | 93.07 | 46.88 | 69.47 |
| QK-PSOFT Full-Right | 290,304 | 68.99 | 93.04 | 45.68 | 69.24 |
| QK-PSOFT Full-Bilateral | 580,608 | 68.35 | 93.76 | 46.17 | 69.43 |
| **QK-BOFT Full-Left** | **69,120** | 69.52 | 91.98 | 44.66 | 68.72 |

Relative to QK-PSOFT Full-Left, QK-BOFT changes accuracy by +1.06 points on
DTD, -1.09 points on EuroSAT, and -2.22 points on dSprites Orientation.  Its
mean is 0.75 points lower.  The parameter reduction is therefore inexpensive
on DTD but too restrictive for EuroSAT and especially dSprites Orientation.

## Rotation diagnostics

The normalized distance from identity is

\[
d(R,I)=\frac{\lVert R-I\rVert_F}{\sqrt{64}}.
\]

| Task | Mean block distance at best epoch | Mean composed distance at best epoch | Mean composed distance at final epoch |
| --- | ---: | ---: | ---: |
| DTD | 0.07 | 0.23 | 0.31 |
| EuroSAT | 0.07 | 0.25 | 0.30 |
| dSprites Orientation | 0.09 | 0.30 | 0.36 |

The learned factors move materially away from identity on every task.  The
weaker dSprites result is therefore not caused by a dead parameterization or
missing gradients.  Instead, dSprites asks for the largest composed movement,
which is consistent with the five-factor, block-size-four structure becoming
a capacity bottleneck.

## Conclusions

1. Structured BOFT is a valid replacement for the dense left spectral
   rotation: it initializes exactly at the pretrained operator, receives
   gradients, and preserves the complete 64-dimensional singular spectrum.
2. It reduces Joint-QK rotation parameters by 76.2% while remaining far above
   Linear Probe on all three tasks.
3. Parameter efficiency is task-dependent.  BOFT improves DTD despite using
   fewer parameters, but loses accuracy on EuroSAT and dSprites Orientation.
4. The result creates a useful capacity--accuracy point rather than a strict
   improvement over dense Joint-QK PSOFT or the existing LoRA and Block-OFT
   baselines.
5. Because the experiment uses one split and one seed, repeated seeds are
   needed before treating sub-point differences as statistically reliable.

The next controlled ablation should vary only BOFT capacity: compare block
sizes 4 and 8, or five and six factors, while keeping the optimizer and data
protocol fixed.  That test can determine whether the dSprites loss comes from
small local blocks, insufficient mixing depth, or both.

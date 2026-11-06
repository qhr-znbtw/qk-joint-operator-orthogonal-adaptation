# QK-PSOFT Full-Rank Left-Rotation Results

## Research question

This experiment tests whether a PSOFT-style rotation can adapt the joint
query-key interaction of each ViT attention head instead of adapting the
query and key weights independently.

For every layer and head, the frozen pretrained operator is

\[
M_h = \widetilde W_{Q,h}\widetilde W_{K,h}^{\mathsf T}
    = U_h\Sigma_hV_h^{\mathsf T},
\]

where the augmented matrices include the query and key biases.  The first
experiment applies a trainable dense left rotation,

\[
M'_h = U_hR_h\Sigma_hV_h^{\mathsf T},
\qquad R_h\in SO(64),
\]

to all 12 heads in all 12 Transformer blocks.  Each rotation is represented
by a zero-initialized skew-symmetric matrix and the scaled Cayley map, so the
initial model exactly reproduces the pretrained model.

## Protocol

- Backbone: ImageNet-pretrained ViT-B/16
- Tasks: Mini-VTAB-3 development split (800 train / 200 validation)
- Seed: 42
- Epochs: 100, with 10 epochs of linear warmup and cosine decay
- Optimizer: AdamW
- Rotation learning rate: `5e-4`
- Classifier learning rate: `1e-3`
- Weight decay: `1e-4`
- Train/evaluation batch size: 64 / 128
- Model selection: highest validation accuracy
- Rotation rank: 64 (the complete compact spectrum)
- Adapted heads: 144
- Rotation parameters: 290,304

The training budget and data protocol are identical to the existing
Linear Probe, LoRA, and block-OFT baselines.

## Test results

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13 | 89.24 | 24.48 | 60.28 |
| LoRA | 69.68 | **94.24** | **49.58** | **71.17** |
| Block-OFT | **70.27** | 93.76 | 46.07 | 70.03 |
| QK-PSOFT Full-Left | 68.46 | 93.07 | 46.88 | 69.47 |

Bold values mark the best result in each column.

Relative to Linear Probe, QK-PSOFT improves DTD by 1.33 points, EuroSAT by
3.83 points, and dSprites Orientation by 22.40 points.  Relative to
Block-OFT, it is lower by 1.81 points on DTD and 0.69 points on EuroSAT, but
higher by 0.81 points on dSprites Orientation.  Its three-task mean is 0.56
points below Block-OFT and 1.70 points below LoRA.

## Cost and model selection

| Task | Best epoch | Best validation accuracy | Test accuracy | Training time | Peak VRAM | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 46 | 66.50 | 68.46 | 319.40 s | 5.34 GiB | 326,447 |
| EuroSAT | 62 | 92.50 | 93.07 | 284.43 s | 5.33 GiB | 297,994 |
| dSprites Orientation | 24 | 50.50 | 46.88 | 280.56 s | 5.34 GiB | 302,608 |

Total training time for the three tasks was 884.39 seconds.  The trainable
parameter count varies only because each task has a different classifier;
the rotation component is always 290,304 parameters.

## Rotation diagnostics

The normalized distance is

\[
d(R,I)=\frac{\lVert R-I\rVert_F}{\sqrt{64}}.
\]

| Task | Distance at best epoch (mean / max) | Distance at final epoch (mean / max) |
| --- | ---: | ---: |
| DTD | 0.250 / 0.300 | 0.268 / 0.316 |
| EuroSAT | 0.190 / 0.240 | 0.190 / 0.240 |
| dSprites Orientation | 0.180 / 0.240 | 0.230 / 0.330 |

All rotations moved away from identity while remaining numerically stable.
dSprites reached its best validation result early and then overfit: the
training accuracy approached 100%, while validation accuracy declined.  This
is direct evidence that a more expressive two-sided variant should not be
assumed to generalize better without regularization or rank control.

## Interpretation

The experiment establishes feasibility: the joint per-head QK operator can
be decomposed, rotated, trained end-to-end, and merged back into ordinary
query/key projections without changing the attention interface.  The method
provides a large gain over Linear Probe and is competitive with block-OFT,
especially on the structured dSprites task, but it does not yet outperform
the strongest baseline consistently.

This is therefore a useful first result rather than a final performance
claim.  The next controlled comparison should keep this protocol fixed and
change only the rotation side:

1. right-only full-rank rotation;
2. two-sided full-rank rotation;
3. only after that, rank reduction or explicit regularization.

Because the current result uses one data split and one seed, additional seeds
will be required before making a statistical claim.

# Internal–External Bilateral Joint-QK BOFT

## Method

For every Transformer layer \(\ell\) and attention head \(h\), the pretrained
bias-aware joint operator is

\[
M_{\ell,h}=U_{\ell,h}\Sigma_{\ell,h}V_{\ell,h}^{\mathsf T}.
\]

The experiment keeps the singular spectrum fixed and combines per-head
compact BOFT rotations with two globally shared ambient BOFT rotations:

\[
\boxed{
M_{\ell,h}'=
\bar Q_LU_{\ell,h}R_{L,\ell,h}\Sigma_{\ell,h}
R_{R,\ell,h}^{\mathsf T}V_{\ell,h}^{\mathsf T}\bar Q_R^{\mathsf T}
}.
\]

Here \(R_{L,\ell,h},R_{R,\ell,h}\in O(64)\) are independent for all 144
heads.  The matrices \(Q_L,Q_R\in O(768)\) are independent from one another
but shared across all heads and all 12 layers.  Since the decomposition
includes query/key biases through one homogeneous coordinate, the actual
ambient operators are

\[
\bar Q_L=\operatorname{diag}(Q_L,1),\qquad
\bar Q_R=\operatorname{diag}(Q_R,1),
\]

so the bias coordinate is not rotated.

The internal BOFT uses block size 4 and five factors.  The external BOFT uses
block size 24 and six butterfly factors, giving full 768-dimensional mixing.
All Cayley coordinates start at zero, hence every rotation is initially the
identity and the initialized network exactly reproduces the pretrained model.

## Parameters and optimization

| Component | Parameters | Learning rate |
| --- | ---: | ---: |
| Per-head internal left/right BOFT | 138,240 | \(5\times10^{-4}\) |
| Global external left/right BOFT | 105,984 | \(10^{-4}\) |
| Total adapter | 244,224 | — |

The singular values are fixed.  The classifier uses learning rate
\(10^{-3}\), and every other setting matches the existing 100-epoch
Mini-VTAB-3 joint-QK protocol.

## Results

| Method | Adapter params | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Internal bilateral QK-BOFT | 138,240 | **69.89%** | 92.91% | **47.11%** | **69.97%** |
| Internal + global external QK-BOFT | 244,224 | 68.88% | **93.22%** | 46.74% | 69.61% |

The shared external rotations change DTD, EuroSAT, and dSprites by -1.01,
+0.31, and -0.37 percentage points.  The mean decreases by 0.36 points.

## Run details

| Task | Best epoch | Best validation | Test accuracy | Training time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| DTD | 17 | 65.50% | 68.88% | 515.64 s | 5.86 GiB |
| EuroSAT | 59 | 95.00% | 93.22% | 461.43 s | 5.86 GiB |
| dSprites Orientation | 34 | 49.50% | 46.74% | 458.26 s | 5.86 GiB |

The complete study required 1,435.33 seconds (23 minutes 55 seconds).

## External-rotation diagnostics

The reported external distance is

\[
d(Q,I)=\frac{\|Q-I\|_F}{\sqrt{768}}.
\]

| Task | Left distance | Right distance | Left gradient norm | Right gradient norm |
| --- | ---: | ---: | ---: | ---: |
| DTD | 0.0993 | 0.0958 | 0.3121 | 0.3054 |
| EuroSAT | 0.1288 | 0.1242 | 0.0097 | 0.0099 |
| dSprites Orientation | 0.1186 | 0.1223 | 1.0210 | 1.1198 |

The external matrices receive nonzero gradients and move measurably away from
identity, so the negative aggregate result is not caused by a disconnected or
inactive adapter.

## Interpretation

1. Moving the ambient left/right subspaces is technically feasible and stable,
   but a single global rotation shared by every layer and head is not
   consistently beneficial.
2. EuroSAT gains 0.31 points, suggesting that a coherent global change of
   feature coordinates can help a specialized task.  DTD and dSprites instead
   need task- or layer-dependent subspace changes.
3. The adapter adds 76.7% more parameters than internal bilateral BOFT while
   reducing the mean by 0.36 points, so it is not a better final method in its
   present form.
4. Both DTD and dSprites reach their best validation checkpoints early and
   then overfit.  More external capacity is therefore unlikely to be the right
   immediate modification.
5. The next controlled structural test should replace one global pair with
   layer-specific but lower-capacity external rotations, or use a low-rank
   Grassmann update that allows different layers to move in different
   directions without introducing 12 full external BOFT pairs.

All reported differences are from one seed (42); sub-one-point differences
need multi-seed confirmation.

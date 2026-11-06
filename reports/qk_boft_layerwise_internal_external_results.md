# Layer-Wise External + Head-Wise Internal Bilateral QK-BOFT

## Method

The global-sharing experiment used one ambient rotation pair for every layer
and head.  This follow-up gives each Transformer layer its own pair while the
12 heads inside that layer share it:

\[
\boxed{
M_{\ell,h}'=
\bar Q_L^{(\ell)}U_{\ell,h}R_{L,\ell,h}\Sigma_{\ell,h}
R_{R,\ell,h}^{\mathsf T}V_{\ell,h}^{\mathsf T}
\bar Q_R^{(\ell)\mathsf T}
}.
\]

Here \(Q_L^{(\ell)},Q_R^{(\ell)}\in O(768)\) are independent across the 12
layers and shared across the 12 heads within a layer.  The compact
\(R_{L,\ell,h},R_{R,\ell,h}\in O(64)\) remain head-specific.  The homogeneous
bias coordinate is fixed through \(\bar Q=\operatorname{diag}(Q,1)\), and all
singular values remain frozen.

The internal rotations use block size 4 and five factors.  Every external
rotation uses block size 24 and six factors.  Zero Cayley coordinates make the
complete model exactly equal to the pretrained network at initialization.

## Parameters

| Component | Parameters | Learning rate |
| --- | ---: | ---: |
| Head-wise internal bilateral BOFT | 138,240 | \(5\times10^{-4}\) |
| Layer-wise external bilateral BOFT | 1,271,808 | \(10^{-4}\) |
| Total adapter | 1,410,048 | — |

The external count is 12 times the global-sharing external count.  The total
adapter has 10.2 times as many parameters as internal-only bilateral QK-BOFT.

## Results

| Method | Adapter params | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Internal bilateral QK-BOFT | 138,240 | **69.89%** | 92.91% | 47.11% | **69.97%** |
| Internal + global external | 244,224 | 68.88% | 93.22% | 46.74% | 69.61% |
| Internal + layer-wise external | 1,410,048 | 68.51% | **93.74%** | **47.14%** | 69.80% |

Changing from global to layer-wise sharing changes DTD, EuroSAT, and dSprites
by -0.37, +0.52, and +0.40 points, raising the mean by 0.19 points.  Compared
with internal-only BOFT, the layer-wise method changes the tasks by -1.38,
+0.83, and +0.03 points and lowers the mean by 0.17 points.

## Run details

| Task | Best epoch | Best validation | Test accuracy | Training time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| DTD | 28 | 66.00% | 68.51% | 509.68 s | 5.91 GiB |
| EuroSAT | 32 | 96.50% | 93.74% | 461.37 s | 5.91 GiB |
| dSprites Orientation | 40 | 50.00% | 47.14% | 458.73 s | 5.91 GiB |

The three runs required 1,429.78 seconds (23 minutes 50 seconds).

## Layer-activity diagnostics

Every layer receives nonzero left and right external gradients.  At the
selected checkpoints, the per-layer external Cayley-parameter norms span:

| Task | Left layer range | Right layer range | Total left gradient | Total right gradient |
| --- | ---: | ---: | ---: | ---: |
| DTD | 0.795–1.402 | 0.870–1.337 | 0.076 | 0.076 |
| EuroSAT | 0.512–1.119 | 0.578–1.071 | 0.012 | 0.013 |
| dSprites Orientation | 0.957–1.525 | 1.031–1.140 | 0.190 | 0.199 |

The layer ranges confirm that the 12 external pairs do not collapse to one
common solution.  The method genuinely learns layer-dependent movements.

## Interpretation

1. The user's sharing hypothesis is supported: allowing each layer to move its
   subspaces independently improves the mean over global sharing by 0.19
   points and improves two of the three tasks.
2. Layer-wise movement is particularly helpful for EuroSAT (+0.83 over
   internal-only BOFT), suggesting that its optimal QK feature coordinates
   differ substantially across representation depth.
3. DTD loses 1.38 points and dominates the aggregate result.  All three tasks
   fit the 800 training examples almost perfectly, so adding 1.27 million
   external parameters mainly increases overfitting rather than universal
   transfer capacity.
4. The method is not yet parameter-efficient: it uses more parameters than
   standard BOFT and remains below both standard BOFT and internal-only QK-BOFT
   in mean accuracy.
5. The appropriate next step is not more external capacity.  It is to retain
   layer-wise independence while reducing each external rotation, for example
   block size 6 with eight factors (506,880 total internal + external adapter
   parameters) or a low-rank Grassmann movement.

All results use seed 42; differences below one point require multi-seed
confirmation.

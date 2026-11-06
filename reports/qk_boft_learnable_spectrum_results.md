# Bilateral Joint-QK BOFT Spectrum Ablation

## Method

Starting from each bias-aware attention-head operator

\[
M_h=U_h\operatorname{diag}(\sigma_h^{(0)})V_h^{\mathsf T},
\]

the experiment learns two independent BOFT rotations and one spectral
log-scale vector:

\[
M_h'=U_hR_{L,h}\operatorname{diag}(\sigma_h')
R_{R,h}^{\mathsf T}V_h^{\mathsf T}.
\]

Two parameterizations are compared.  The original norm-preserving variant
uses centered log-scales and restores the original spectral L2 norm:

\[
\beta_h=\alpha_h-\operatorname{mean}(\alpha_h),
\qquad
\widetilde\sigma_h=\sigma_h^{(0)}\odot\exp(\beta_h).
\]

The final spectrum restores the original per-head L2 norm:

\[
\boxed{
\sigma_h'=\widetilde\sigma_h
\frac{\|\sigma_h^{(0)}\|_2}{\|\widetilde\sigma_h\|_2}
}.
\]

This parameterization preserves positivity, exactly reproduces the pretrained
spectrum at initialization, removes the redundant common log-scale direction,
and learns spectral shape without changing total spectral energy.

The new unconstrained variant removes both normalization operations:

\[
\boxed{
\sigma_h'=\sigma_h^{(0)}\odot\exp(\alpha_h)
}.
\]

It remains positive and is still exactly the pretrained spectrum when
\(\alpha_h=0\), but it can now learn both relative spectral shape and a common
scale (attention-temperature) direction.

There are \(144\times64=9{,}216\) learned spectrum parameters.  Together with
138,240 bilateral BOFT parameters, the adapter has 147,456 parameters.  BOFT,
spectrum, and classifier learning rates are \(5\times10^{-4}\),
\(10^{-4}\), and \(10^{-3}\), respectively.  Every other experimental setting
matches the fixed-spectrum bilateral BOFT run.

## Results

| Method | Adapter parameters | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bilateral QK-BOFT, fixed spectrum | 138,240 | **69.89** | 92.91 | **47.11** | **69.97** |
| Bilateral QK-BOFT, norm-preserving spectrum | 147,456 | 68.35 | **93.07** | 47.05 | 69.49 |
| Bilateral QK-BOFT, unconstrained spectrum | 147,456 | 69.15 | 92.94 | 46.92 | 69.67 |

Relative to the normalized version, removing normalization changes accuracy
by +0.80 points on DTD, -0.13 on EuroSAT, and -0.13 on dSprites Orientation,
raising the mean by 0.18 points.  Relative to fixed-spectrum BOFT, however, the
new version changes the three tasks by -0.74, +0.03, and -0.19 points and lowers
the mean by 0.30 points.  The added freedom therefore recovers part of the old
variant's DTD loss but does not provide an aggregate improvement.

## Run details

| Task | Best epoch | Best validation | Test accuracy | Training time | Peak VRAM | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | 25 | 66.00 | 69.15 | 451.01 s | 5.73 GiB | 183,599 |
| EuroSAT | 47 | 92.50 | 92.94 | 436.77 s | 5.73 GiB | 155,146 |
| dSprites Orientation | 45 | 47.50 | 46.92 | 423.82 s | 5.73 GiB | 159,760 |

The three new runs required 1,311.60 seconds.  Peak VRAM was 5.73 GiB.  The
adapter still contains 138,240 BOFT parameters and 9,216 spectrum parameters;
removing normalization does not alter parameter count.

## Learned-spectrum diagnostics at the selected checkpoint

The relative per-head spectrum change is

\[
d_\sigma=
\frac{\|\sigma_h'-\sigma_h^{(0)}\|_2}
{\|\sigma_h^{(0)}\|_2}.
\]

| Task | Mean relative change | Maximum relative change | Smallest value ratio | Largest value ratio |
| --- | ---: | ---: | ---: | ---: |
| DTD | 0.72% | 1.27% | 0.978 | 1.024 |
| EuroSAT | 0.74% | 1.22% | 0.978 | 1.028 |
| dSprites Orientation | 0.97% | 2.13% | 0.971 | 1.046 |

All tasks receive nonzero spectrum gradients, and the selected checkpoints
remain close to the original spectrum even without an explicit norm
constraint.  Thus the optimizer does not exploit the newly available common
scale aggressively at the current spectrum learning rate of \(10^{-4}\).

## Interpretation

1. The unconstrained parameterization is valid and stable.  It preserves
   positivity, initializes exactly at the pretrained model, and receives
   nonzero gradients without numerical instability.
2. Removing spectral normalization is mildly beneficial relative to the
   norm-preserving variant (+0.18 mean), mostly because DTD recovers 0.80
   points.
3. Fixed singular values are still not the main observed bottleneck: adding
   9,216 spectrum parameters remains 0.30 points below fixed-spectrum BOFT.
4. The learned spectrum moves by less than 1% on average at every selected
   checkpoint.  The current \(10^{-4}\) learning rate makes this a conservative
   test of unconstrained scaling rather than a large spectral deformation.
5. Because every result uses only seed 42, changes below one point should be
   treated as inconclusive until repeated across seeds.

The most controlled follow-up is now a spectrum-learning-rate ablation at
\(10^{-4}\), \(3\times10^{-4}\), and \(5\times10^{-4}\), preferably with three
seeds.  That experiment would distinguish "unconstrained scale is not useful"
from "the optimizer barely moved the unconstrained scale."

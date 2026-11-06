# Joint QK Spectrum Analysis for ViT-B/16

## Scope

The analysis covers all 12 Transformer layers and all 12 heads per layer,
for a total of 144 bias-aware joint query-key operators.  Each augmented
operator has shape `769 x 769` and compact rank at most 64.

For head \(h\), the cumulative Frobenius-energy ratio at rank \(r\) is

\[
E_h(r)=
\frac{\sum_{i=1}^{r}\sigma_{h,i}^{2}}
     {\sum_{i=1}^{64}\sigma_{h,i}^{2}}.
\]

## Aggregate Energy

| Rank | Minimum | P25 | Median | Mean | P75 | Maximum | Heads >= 90% | Heads >= 95% | Heads >= 99% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.125 | 0.185 | 0.235 | 0.297 | 0.346 | 0.984 | 1.4% | 0.7% | 0.0% |
| 8 | 0.222 | 0.298 | 0.371 | 0.434 | 0.532 | 0.997 | 4.2% | 3.5% | 0.7% |
| 12 | 0.310 | 0.395 | 0.480 | 0.533 | 0.648 | 0.999 | 5.6% | 4.2% | 1.4% |
| 16 | 0.390 | 0.479 | 0.569 | 0.613 | 0.740 | 1.000 | 9.7% | 5.6% | 2.8% |
| 24 | 0.532 | 0.620 | 0.708 | 0.734 | 0.853 | 1.000 | 18.1% | 11.1% | 4.9% |
| 32 | 0.653 | 0.736 | 0.817 | 0.823 | 0.924 | 1.000 | 30.6% | 18.8% | 10.4% |
| 40 | 0.760 | 0.832 | 0.895 | 0.891 | 0.969 | 1.000 | 45.8% | 30.6% | 17.4% |
| 48 | 0.852 | 0.908 | 0.949 | 0.942 | 0.991 | 1.000 | 75.7% | 49.3% | 25.7% |
| 56 | 0.934 | 0.965 | 0.985 | 0.978 | 0.999 | 1.000 | 100.0% | 83.3% | 38.9% |
| 64 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% |

The median head requires rank 41, 49, and 58 to reach 90%, 95%, and 99%
energy respectively.  The mean required ranks are 37.8, 44.1, and 52.6.

## Layer Dependence

| Layer | r=16 | r=24 | r=32 | r=40 | r=48 | r=56 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.898 | 0.949 | 0.980 | 0.994 | 0.999 | 1.000 |
| 2 | 0.894 | 0.965 | 0.988 | 0.997 | 1.000 | 1.000 |
| 3 | 0.758 | 0.872 | 0.937 | 0.974 | 0.992 | 0.999 |
| 4 | 0.708 | 0.823 | 0.898 | 0.948 | 0.978 | 0.994 |
| 5 | 0.643 | 0.773 | 0.862 | 0.925 | 0.967 | 0.992 |
| 6 | 0.593 | 0.730 | 0.830 | 0.903 | 0.954 | 0.987 |
| 7 | 0.564 | 0.703 | 0.808 | 0.886 | 0.944 | 0.983 |
| 8 | 0.529 | 0.668 | 0.777 | 0.863 | 0.928 | 0.975 |
| 9 | 0.498 | 0.640 | 0.754 | 0.846 | 0.917 | 0.971 |
| 10 | 0.447 | 0.591 | 0.711 | 0.811 | 0.893 | 0.958 |
| 11 | 0.400 | 0.543 | 0.667 | 0.774 | 0.866 | 0.943 |
| 12 | 0.421 | 0.554 | 0.670 | 0.773 | 0.862 | 0.939 |

The spectrum becomes progressively flatter with depth.  A rank that is nearly
lossless for the first two layers can omit a substantial fraction of the
operator energy in the final layers.  This is evidence against choosing the
rank from a single layer or a single representative head.

## Rank Decision

The first controlled study will use one uniform rank across all 144 heads.
The proposed ranks are:

| Role | Rank | Mean energy | Median energy | Single-sided adapter parameters |
| --- | ---: | ---: | ---: | ---: |
| Low-budget ablation | 32 | 82.3% | 81.7% | 71,424 |
| Efficiency ablation | 48 | 94.2% | 94.9% | 162,432 |
| Primary setting | 56 | 97.8% | 98.5% | 221,760 |

Rank 56 is the smallest tested multiple of eight whose median energy exceeds
95%.  It also guarantees at least 90% energy for every head while retaining an
eight-dimensional frozen residual.  Rank 64 is useful as a full-spectrum
control, but it removes the principal/residual distinction and is therefore not
the primary PSOFT setting.

For later bilateral experiments, rank 40 uses 224,640 rotation parameters and
is approximately parameter-matched to the rank-56 single-sided setting.

## Reproduction

Run:

```bash
python -m scripts.analyze_qk_spectrum
```

The script loads the local pretrained ViT-B/16 checkpoint and recomputes all
statistics directly from the 144 compact QK decompositions.

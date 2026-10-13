# ODES reproduction and FreqODES results (CHAOS T1 IP -> OOP, b = 1%)

## Table 1 - Dice (%) per organ

| Method | Liver | L.Kidney | R.Kidney | Spleen | Mean | Mean Jaccard | Annotated px % | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Source only | 85.89 | 29.84 | 33.74 | 70.43 | 54.97 | 42.74 | 0.000 | 49 |
| ODES K=100 (reproduced) | 87.89 | 76.90 | 75.28 | 78.00 | 79.52 | 66.96 | 0.992 | 117 |
| ODES (BN-KL) K=50 | 88.02 | 76.66 | 75.71 | 77.89 | 79.57 | 67.00 | 0.500 | 171 |
| FreqODES-FFT K=50 | 88.08 | 76.43 | 75.08 | 77.37 | 79.24 | 66.61 | 0.500 | 115 |
| FreqODES-DWT K=50 | 87.97 | 76.35 | 74.62 | 77.41 | 79.09 | 66.43 | 0.500 | 114 |
| FreqODES-FFT+DWT K=50 | 87.84 | 76.38 | 74.79 | 77.80 | 79.20 | 66.52 | 0.500 | 114 |
| FreqODES-Fused (lambda=0.5) K=50 | 88.04 | 76.70 | 75.27 | 77.99 | 79.50 | 66.92 | 0.500 | 173 |
| Random pruning K=50 | 87.81 | 77.41 | 75.26 | 77.59 | 79.52 | 67.00 | 0.500 | 113 |
| ODES (BN-KL) K=10 | 87.88 | 75.19 | 74.18 | 78.06 | 78.83 | 66.24 | 0.109 | 167 |
| FreqODES-FFT K=10 | 87.43 | 73.65 | 73.59 | 77.41 | 78.02 | 65.03 | 0.109 | 107 |
| FreqODES-DWT K=10 | 87.45 | 74.00 | 73.31 | 77.84 | 78.15 | 65.28 | 0.109 | 109 |
| FreqODES-FFT+DWT K=10 | 87.28 | 74.31 | 73.46 | 77.74 | 78.20 | 65.27 | 0.109 | 109 |
| FreqODES-Fused (lambda=0.5) K=10 | 86.11 | 75.22 | 74.15 | 78.12 | 78.40 | 65.45 | 0.109 | 169 |
| Random pruning K=10 | 88.11 | 75.80 | 74.34 | 76.75 | 78.75 | 66.04 | 0.109 | 107 |
| Paper: Source only | 87.77 | 37.97 | 18.92 | 67.31 | 52.99 | - | - | - |
| Paper: ODES K=100 | 88.90 | 72.41 | 74.55 | 78.56 | 78.61 | - | - | - |
| Paper: ODES K=50 | 89.13 | 72.30 | 74.43 | 78.36 | 78.56 | - | - | - |
| Paper: ODES K=10 | 88.41 | 71.45 | 70.22 | 77.12 | 76.80 | - | - | - |

## Table 3 - Image pruning x Diversity weighting (K = 10)

| Pruning | D.W. off | D.W. on |
|---|---|---|
| BN-KL (ODES) | 78.73 | 78.83 |
| FFT+DWT (FreqODES) | 78.32 | 78.20 |
| Random | 78.96 | 78.75 |

## Fusion weight sweep (K = 10): D = lambda*BN + (1-lambda)*Freq

| lambda | Mean Dice |
|---|---|
| 0 (pure FFT+DWT) | 78.20 |
| 0.25 | 79.09 |
| 0.5 | 78.40 |
| 0.75 | 78.93 |
| 1 (pure BN, ODES) | 78.83 |

## Paired Wilcoxon vs ODES (BN-KL) over 20 patients

| Run | K | Mean diff (Dice) | p-value |
|---|---|---|---|
| FFT_K0.1 | 0.1 | -0.81 | 0.005 |
| DWT_K0.1 | 0.1 | -0.67 | 0.027 |
| FREQ_K0.1 | 0.1 | -0.63 | 0.091 |
| FUSED_K0.1_lam0.5 | 0.1 | -0.43 | 0.159 |
| RANDOM_K0.1 | 0.1 | -0.08 | 0.334 |
| FFT_K0.5 | 0.5 | -0.33 | 0.064 |
| DWT_K0.5 | 0.5 | -0.48 | 0.030 |
| FREQ_K0.5 | 0.5 | -0.37 | 0.044 |
| FUSED_K0.5_lam0.5 | 0.5 | -0.07 | 0.687 |
| RANDOM_K0.5 | 0.5 | -0.05 | 1.000 |

## Shift-detector separability (held-out IP vs OOP slices)

| Descriptor | ROC-AUC | Mean score IP | Mean score OOP |
|---|---|---|---|
| FFT | 0.994 | 2.56 | 21.32 |
| DWT | 0.969 | 2.04 | 13.39 |
| FREQ | 0.993 | 2.30 | 17.35 |

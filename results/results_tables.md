# ODES reproduction and FreqODES results (CHAOS T1 IP -> OOP, b = 1%)

## Table 1 - Dice (%) per organ

| Method | Liver | L.Kidney | R.Kidney | Spleen | Mean | Annotated px % | Time (s) |
|---|---|---|---|---|---|---|---|
| Source only | 84.17 | 47.28 | 52.94 | 72.76 | 64.29 | 0.000 | 53 |
| ODES K=100 (reproduced) | 87.74 | 77.28 | 74.72 | 77.32 | 79.26 | 0.992 | 125 |
| ODES (BN-KL) K=50 | 87.82 | 76.99 | 74.71 | 77.51 | 79.26 | 0.500 | 181 |
| FreqODES-FFT K=50 | 87.89 | 77.39 | 74.63 | 77.43 | 79.34 | 0.500 | 123 |
| FreqODES-DWT K=50 | 87.69 | 76.68 | 74.16 | 77.34 | 78.97 | 0.500 | 120 |
| FreqODES-FFT+DWT K=50 | 87.76 | 76.67 | 73.96 | 77.24 | 78.91 | 0.500 | 123 |
| FreqODES-Fused (lambda=0.5) K=50 | 87.84 | 77.29 | 74.59 | 77.68 | 79.35 | 0.500 | 183 |
| Random pruning K=50 | 87.61 | 77.26 | 74.31 | 77.67 | 79.21 | 0.500 | 120 |
| ODES (BN-KL) K=10 | 87.70 | 75.59 | 73.51 | 76.71 | 78.38 | 0.109 | 176 |
| FreqODES-FFT K=10 | 86.69 | 73.43 | 72.21 | 76.93 | 77.31 | 0.109 | 116 |
| FreqODES-DWT K=10 | 86.92 | 74.47 | 71.95 | 75.99 | 77.33 | 0.109 | 116 |
| FreqODES-FFT+DWT K=10 | 86.79 | 74.53 | 72.26 | 77.03 | 77.65 | 0.109 | 117 |
| FreqODES-Fused (lambda=0.5) K=10 | 87.03 | 75.41 | 72.97 | 77.19 | 78.15 | 0.109 | 178 |
| Random pruning K=10 | 87.49 | 75.36 | 73.33 | 76.44 | 78.15 | 0.109 | 115 |
| Paper: Source only | 87.77 | 37.97 | 18.92 | 67.31 | 52.99 | - | - |
| Paper: ODES K=100 | 88.90 | 72.41 | 74.55 | 78.56 | 78.61 | - | - |
| Paper: ODES K=50 | 89.13 | 72.30 | 74.43 | 78.36 | 78.56 | - | - |
| Paper: ODES K=10 | 88.41 | 71.45 | 70.22 | 77.12 | 76.80 | - | - |

## Table 3 - Image pruning x Diversity weighting (K = 10)

| Pruning | D.W. off | D.W. on |
|---|---|---|
| BN-KL (ODES) | 78.48 | 78.38 |
| FFT+DWT (FreqODES) | 77.30 | 77.65 |
| Random | 78.19 | 78.15 |

## Fusion weight sweep (K = 10): D = lambda*BN + (1-lambda)*Freq

| lambda | Mean Dice |
|---|---|
| 0 (pure FFT+DWT) | 77.65 |
| 0.25 | 78.11 |
| 0.5 | 78.15 |
| 0.75 | 77.98 |
| 1 (pure BN, ODES) | 78.38 |

## Paired Wilcoxon vs ODES (BN-KL) over 20 patients

| Run | K | Mean diff (Dice) | p-value |
|---|---|---|---|
| FFT_K0.1 | 0.1 | -1.06 | 0.009 |
| DWT_K0.1 | 0.1 | -1.05 | 0.005 |
| FREQ_K0.1 | 0.1 | -0.73 | 0.044 |
| FUSED_K0.1_lam0.5 | 0.1 | -0.23 | 0.469 |
| RANDOM_K0.1 | 0.1 | -0.22 | 0.573 |
| FFT_K0.5 | 0.5 | +0.08 | 0.748 |
| DWT_K0.5 | 0.5 | -0.29 | 0.136 |
| FREQ_K0.5 | 0.5 | -0.35 | 0.070 |
| FUSED_K0.5_lam0.5 | 0.5 | +0.09 | 0.629 |
| RANDOM_K0.5 | 0.5 | -0.05 | 0.904 |

## Shift-detector separability (held-out IP vs OOP slices)

| Descriptor | ROC-AUC | Mean score IP | Mean score OOP |
|---|---|---|---|
| FFT | 0.994 | 2.56 | 21.32 |
| DWT | 0.969 | 2.04 | 13.39 |
| FREQ | 0.993 | 2.30 | 17.35 |

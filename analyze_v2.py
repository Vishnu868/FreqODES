"""Analyse the FreqODES v2 experiment (3 seeds).
    python analyze_v2.py results
Prints mean +- std over seeds, per-seed Dice, and a paired Wilcoxon test vs ODES (BN)
over all patient x seed pairs (20 patients x 3 seeds = 60 pairs). Writes results/v2_summary.json.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

src = sys.argv[1]
R = {}
for f in glob.glob(os.path.join(src, 'json', 'V2_*_run_0.json')):
    n = os.path.basename(f)[3:-11]                  # e.g. FREQV2_K0.1_s2
    m, K, s = n.rsplit('_', 2)
    R[(m, K[1:], s[1:])] = json.load(open(f))
seeds = sorted({k[2] for k in R}, key=int)
methods = ['BN', 'FREQ', 'FREQV2', 'FREQAREA', 'FREQENT', 'RANDOM']
label = {'BN': 'ODES (BN-KL)', 'FREQ': 'FreqODES v1 (FFT+DWT)', 'FREQV2': 'FreqODES v2 (freq x area x entropy)',
         'FREQAREA': 'Ablation: freq x area', 'FREQENT': 'Ablation: freq x entropy', 'RANDOM': 'Random'}
out = {'seeds': seeds, 'rows': []}
for K in ('0.1', '0.5'):
    print(f'\n=== K = {float(K) * 100:.0f}%  (seeds: {", ".join(seeds)})')
    print(f'{"method":40s} {"mean":>7s} {"std":>6s}  ' + '  '.join(f's{s:>5s}' for s in seeds) + f'  {"diff":>6s} {"p":>6s} {"IoU":>6s} {"time":>5s}')
    for m in methods:
        if not all((m, K, s) in R for s in seeds):
            continue
        d = np.array([R[(m, K, s)]['mean_dice'] for s in seeds])
        iou = np.mean([R[(m, K, s)]['mean_jaccard'] for s in seeds])
        t = np.mean([R[(m, K, s)]['seconds'] for s in seeds])
        x = np.concatenate([R[(m, K, s)]['per_batch_mean_dice'] for s in seeds])
        b = np.concatenate([R[('BN', K, s)]['per_batch_mean_dice'] for s in seeds])
        diff, p = (float(np.mean(x - b)), float(wilcoxon(x, b).pvalue)) if m != 'BN' else (0.0, 1.0)
        print(f'{label[m]:40s} {d.mean():7.2f} {d.std(ddof=1):6.2f}  ' + '  '.join(f'{v:6.2f}' for v in d)
              + f'  {diff:+6.2f} {p:6.3f} {iou:6.2f} {t:5.0f}')
        out['rows'].append(dict(method=m, label=label[m], K=K, mean=float(d.mean()), std=float(d.std(ddof=1)),
                                per_seed=d.tolist(), diff_vs_odes=diff, p_vs_odes=p, iou=float(iou), seconds=float(t)))
json.dump(out, open(os.path.join(src, 'v2_summary.json'), 'w'), indent=1)

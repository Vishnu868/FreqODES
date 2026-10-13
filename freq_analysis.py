"""Qualitative / quantitative check of the FreqODES shift detector.

1. Mean FFT radial-band spectrum and DWT sub-band energies: in-phase (source) vs out-of-phase (target).
2. Shift-score distributions + ROC-AUC of separating held-out source slices from target slices.
Writes results/freq_analysis.png and results/freq_analysis.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset

from dataloader.data_loader_stack import Stack_Loader
from methods.freq_shift import FrequencyReference, fft_descriptor, dwt_descriptor

LBL = [63, 126, 189, 252]


def load(path):
    ds = ConcatDataset([Stack_Loader(path, LBL, split=s) for s in ('train', 'test')])
    return [ds[i][0].numpy() for i in range(len(ds))]


src = load('../data/CHAOS/T1DUAL_InPhase')
tgt = load('../data/CHAOS/T1DUAL_OutPhase')
# reference from the first 10 source patients, held-out source = remaining patients
ref_slices = np.concatenate(src[:10])
held_src = np.concatenate(src[10:])
tgt_all = np.concatenate(tgt)
rng = np.random.RandomState(0)
ref = FrequencyReference(ref_slices[rng.choice(len(ref_slices), 200, replace=False)])

out = {}
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for name, fn, a in (('FFT', fft_descriptor, ax[0]), ('DWT', dwt_descriptor, ax[1])):
    s = np.stack([fn(x) for x in held_src]); t = np.stack([fn(x) for x in tgt_all])
    a.errorbar(range(s.shape[1]), s.mean(0), s.std(0), label='In-phase (source)', capsize=2)
    a.errorbar(range(t.shape[1]), t.mean(0), t.std(0), label='Out-of-phase (target)', capsize=2)
    a.set_title(f'{name} descriptor (log relative energy)'); a.legend()
    a.set_xlabel('radial band (low -> high freq)' if name == 'FFT' else 'sub-band: LL3, (LH,HL,HH) L3..L1')
for k, kind in enumerate(('FFT', 'DWT', 'FREQ')):
    ss = ref.score_stack(held_src, kind); ts = ref.score_stack(tgt_all, kind)
    auc = roc_auc_score(np.r_[np.zeros(len(ss)), np.ones(len(ts))], np.r_[ss, ts])
    out[kind] = dict(auc=float(auc), src_mean=float(ss.mean()), tgt_mean=float(ts.mean()))
    if kind == 'FREQ':
        ax[2].hist(np.log(ss), 40, alpha=.6, label='held-out in-phase'); ax[2].hist(np.log(ts), 40, alpha=.6, label='out-of-phase')
        ax[2].set_title(f'log FREQ shift score (AUC={auc:.3f})'); ax[2].legend()
# synthetic-shift sanity check: Gaussian noise / blur of held-out source slices
from scipy.ndimage import gaussian_filter
lv = [0, .01, .02, .05, .1]
ax[3].plot(lv, [ref.score_stack(np.clip(held_src[::5] + rng.randn(*held_src[::5].shape) * l, 0, 1), 'FREQ').mean() for l in lv], 'o-', label='additive noise sigma')
bl = [0, .5, 1, 2, 3]
ax[3].plot(np.array(bl) / 30, [ref.score_stack(np.stack([gaussian_filter(x, b) for x in held_src[::5]]), 'FREQ').mean() for b in bl], 's-', label='blur sigma (x1/30)')
ax[3].set_yscale('log'); ax[3].set_title('Synthetic shift -> FREQ score'); ax[3].legend()
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/freq_analysis.png', dpi=120)
json.dump(out, open('results/freq_analysis.json', 'w'), indent=1)
print(out)

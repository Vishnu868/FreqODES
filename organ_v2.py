r"""Per-organ Dice/IoU and per-patient wins for the 3-seed v2 runs -> LaTeX macros.
    python organ_v2.py results
Writes results/figures/v2_organ_numbers.tex with macros \V<method><K><organ>[J] and \V<method><K>Wins.
"""
import json
import os
import sys

import numpy as np

src = sys.argv[1]
SEEDS = ('1', '2', '12345')
CODE = {'BN': 'Bn', 'FREQ': 'Fq', 'FREQV2': 'Fv', 'FREQAREA': 'Fa', 'FREQENT': 'Fe', 'RANDOM': 'Rd'}
KN = {'0.1': 'Ten', '0.5': 'Fifty'}
NAMES = ['Liver', 'LK', 'RK', 'Spleen']


def load(m, K, s):
    return json.load(open(os.path.join(src, 'json', f'V2_{m}_K{K}_s{s}_run_0.json')))


def nc(name, val):
    return '\\newcommand{\\' + name + '}{' + val + '}'


organs = load('BN', '0.1', SEEDS[0])['organs']
order = [organs.index(o) for o in ('liver', 'left kidney', 'right kidney', 'spleen')]
mac = []
for m, c in CODE.items():
    for K, kn in KN.items():
        try:
            runs = [load(m, K, s) for s in SEEDS]
        except FileNotFoundError:
            continue
        for j, i in enumerate(order):
            mac.append(nc(f'V{c}{kn}{NAMES[j]}', f"{np.mean([r['organ_dice'][i] for r in runs]):.2f}"))
            mac.append(nc(f'V{c}{kn}{NAMES[j]}J', f"{np.mean([r['organ_jaccard'][i] for r in runs]):.2f}"))
        if m != 'BN':
            x = np.concatenate([r['per_batch_mean_dice'] for r in runs])
            b = np.concatenate([load('BN', K, s)['per_batch_mean_dice'] for s in SEEDS])
            mac.append(nc(f'V{c}{kn}Wins', str(int((x > b).sum()))))
open(os.path.join(src, 'figures', 'v2_organ_numbers.tex'), 'w').write('\n'.join(mac) + '\n')
print(len(mac), 'macros')

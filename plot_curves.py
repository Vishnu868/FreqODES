"""Training-loss curve and per-patient online Dice curve.
    python plot_curves.py <results_dir> <out_dir>
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

loss = [float(v) for v in re.findall(r'loss ([0-9.]+)', open(os.path.join(src, 'logs', 'logs_train_source.txt')).read())]
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(range(1, len(loss) + 1), loss, 'o-')
ax.set_xlabel('Epoch'); ax.set_ylabel('CE + Dice loss'); ax.set_title('Source training (CHAOS T1 in-phase)'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(out, 'train_loss.png'), dpi=150)

R = lambda n: json.load(open(os.path.join(src, 'json', f'{n}_run_0.json')))
fig, ax = plt.subplots(figsize=(6, 3.4))
for n, label in [('source_only', 'Source only'), ('BN_K0.1', 'ODES (BN-KL), K=10'),
                 ('FREQ_K0.1', 'FreqODES FFT+DWT, K=10'), ('ODES_K100', 'ODES, K=100')]:
    d = R(n)['per_batch_mean_dice']
    ax.plot(range(1, len(d) + 1), d, 'o-', ms=3, label=label)
ax.set_xlabel('Incoming patient (online batch)'); ax.set_ylabel('Mean Dice (%)')
ax.set_title('Online adaptation: Dice per incoming patient'); ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(out, 'online_dice.png'), dpi=150)

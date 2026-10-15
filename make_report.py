"""Turn results/json/*.json into report tables (Markdown + CSV) and figures.
    python make_report.py <results_dir> <out_dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
R = {os.path.basename(f).replace('_run_0.json', ''): json.load(open(f)) for f in glob.glob(os.path.join(src, 'json', '*.json')) if not os.path.basename(f).startswith('V2_')}
organs = R['source_only']['organs']
# paper column order: liver, L.kidney, R.kidney, spleen
order = [organs.index(o) for o in ('liver', 'left kidney', 'right kidney', 'spleen')]
hdr = ['Liver', 'L.Kidney', 'R.Kidney', 'Spleen', 'Mean']


def row(name, label):
    r = R[name]
    d = [r['organ_dice'][i] for i in order]
    jac = f"{r['mean_jaccard']:.2f}" if 'mean_jaccard' in r else '-'
    return [label] + [f'{x:.2f}' for x in d] + [f"{r['mean_dice']:.2f}", jac, f"{100 * r['annotated_fraction']:.3f}", f"{r['seconds']:.0f}"]


def table(rows, cols):
    s = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    return s + ''.join('| ' + ' | '.join(r) + ' |\n' for r in rows)


cols = ['Method'] + hdr + ['Mean Jaccard', 'Annotated px %', 'Time (s)']
t1 = [row('source_only', 'Source only'), row('ODES_K100', 'ODES K=100 (reproduced)')]
for K in ('0.5', '0.1'):
    t1 += [row(f'BN_K{K}', f'ODES (BN-KL) K={int(float(K)*100)}'),
           row(f'FFT_K{K}', f'FreqODES-FFT K={int(float(K)*100)}'),
           row(f'DWT_K{K}', f'FreqODES-DWT K={int(float(K)*100)}'),
           row(f'FREQ_K{K}', f'FreqODES-FFT+DWT K={int(float(K)*100)}'),
           row(f'FUSED_K{K}_lam0.5', f'FreqODES-Fused (lambda=0.5) K={int(float(K)*100)}'),
           row(f'RANDOM_K{K}', f'Random pruning K={int(float(K)*100)}')]
paper = [['Paper: Source only', '87.77', '37.97', '18.92', '67.31', '52.99', '-', '-', '-'],
         ['Paper: ODES K=100', '88.90', '72.41', '74.55', '78.56', '78.61', '-', '-', '-'],
         ['Paper: ODES K=50', '89.13', '72.30', '74.43', '78.36', '78.56', '-', '-', '-'],
         ['Paper: ODES K=10', '88.41', '71.45', '70.22', '77.12', '76.80', '-', '-', '-']]

t3 = [['BN-KL (ODES)', f"{R['BN_K0.1_noDW']['mean_dice']:.2f}", f"{R['BN_K0.1']['mean_dice']:.2f}"],
      ['FFT+DWT (FreqODES)', f"{R['FREQ_K0.1_noDW']['mean_dice']:.2f}", f"{R['FREQ_K0.1']['mean_dice']:.2f}"],
      ['Random', f"{R['RANDOM_K0.1_noDW']['mean_dice']:.2f}", f"{R['RANDOM_K0.1']['mean_dice']:.2f}"]]
lam = [[l, f"{R[n]['mean_dice']:.2f}"] for l, n in (('0 (pure FFT+DWT)', 'FREQ_K0.1'), ('0.25', 'FUSED_K0.1_lam0.25'), ('0.5', 'FUSED_K0.1_lam0.5'),
                                                    ('0.75', 'FUSED_K0.1_lam0.75'), ('1 (pure BN, ODES)', 'BN_K0.1'))]

# paired Wilcoxon over the 20 patient batches vs ODES BN-KL at same K
stats = []
for K in ('0.1', '0.5'):
    base = np.array(R[f'BN_K{K}']['per_batch_mean_dice'])
    for n in (f'FFT_K{K}', f'DWT_K{K}', f'FREQ_K{K}', f'FUSED_K{K}_lam0.5', f'RANDOM_K{K}'):
        x = np.array(R[n]['per_batch_mean_dice'])
        p = wilcoxon(x, base).pvalue if np.any(x != base) else 1.0
        stats.append([n, K, f'{np.mean(x - base):+.2f}', f'{p:.3f}'])

fa = json.load(open(os.path.join(src, 'freq_analysis.json')))
md = '# ODES reproduction and FreqODES results (CHAOS T1 IP -> OOP, b = 1%)\n\n'
md += '## Table 1 - Dice (%) per organ\n\n' + table(t1 + paper, cols)
md += '\n## Table 3 - Image pruning x Diversity weighting (K = 10)\n\n' + table(t3, ['Pruning', 'D.W. off', 'D.W. on'])
md += '\n## Fusion weight sweep (K = 10): D = lambda*BN + (1-lambda)*Freq\n\n' + table(lam, ['lambda', 'Mean Dice'])
md += '\n## Paired Wilcoxon vs ODES (BN-KL) over 20 patients\n\n' + table(stats, ['Run', 'K', 'Mean diff (Dice)', 'p-value'])
md += '\n## Shift-detector separability (held-out IP vs OOP slices)\n\n' + table([[k, f"{v['auc']:.3f}", f"{v['src_mean']:.2f}", f"{v['tgt_mean']:.2f}"] for k, v in fa.items()],
                                                                                    ['Descriptor', 'ROC-AUC', 'Mean score IP', 'Mean score OOP'])
open(os.path.join(out, 'results_tables.md'), 'w').write(md)
with open(os.path.join(out, 'results_table1.csv'), 'w') as f:
    f.write(','.join(cols) + '\n' + ''.join(','.join(r) + '\n' for r in t1 + paper))

# Figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
methods = [('BN', 'ODES (BN-KL)'), ('FFT', 'FreqODES FFT'), ('DWT', 'FreqODES DWT'), ('FREQ', 'FreqODES FFT+DWT'), ('FUSED', 'FreqODES Fused'), ('RANDOM', 'Random')]
for p, lab in methods:
    ys = [R[f'{p}_K0.1' + ('_lam0.5' if p == 'FUSED' else '')]['mean_dice'], R[f'{p}_K0.5' + ('_lam0.5' if p == 'FUSED' else '')]['mean_dice'], R['ODES_K100']['mean_dice']]
    ax[0].plot([10, 50, 100], ys, 'o-', label=lab)
ax[0].axhline(R['source_only']['mean_dice'], ls='--', c='gray', label='Source only')
ax[0].set_xlabel('K (% slices annotated per patient)'); ax[0].set_ylabel('Mean Dice (%)'); ax[0].set_title('Dice vs annotation budget'); ax[0].legend(fontsize=8)
names = ['BN_K0.1', 'FREQ_K0.1', 'FUSED_K0.1_lam0.5', 'RANDOM_K0.1']
ax[1].boxplot([R[n]['per_batch_mean_dice'] for n in names], labels=['ODES\nBN-KL', 'FreqODES\nFFT+DWT', 'FreqODES\nFused', 'Random'])
ax[1].set_ylabel('Per-patient mean Dice (%)'); ax[1].set_title('Per-patient Dice, K = 10')
plt.tight_layout(); plt.savefig(os.path.join(out, 'dice_vs_budget.png'), dpi=130)

fig, ax = plt.subplots(figsize=(6, 4))
t = [R[n]['seconds'] for n in ('BN_K0.1', 'FREQ_K0.1', 'FFT_K0.1', 'RANDOM_K0.1')]
ax.bar(['ODES\nBN-KL', 'FreqODES\nFFT+DWT', 'FreqODES\nFFT', 'Random'], t, color=['#888', '#2a7', '#2a7', '#ccc'])
ax.set_ylabel('Seconds for full online pass (20 patients, T4)'); ax.set_title('Adaptation cost, K = 10')
plt.tight_layout(); plt.savefig(os.path.join(out, 'runtime.png'), dpi=130)
print(md)


# LaTeX macros with every number used by the slides / report
def tname(n):
    n = n.replace('0.25', 'Quarter').replace('0.75', 'ThreeQ').replace('0.1', 'Ten').replace('0.5', 'Fifty').replace('lam', 'L')
    return ''.join(ch for ch in n if ch.isalpha())


def nc(name, val):
    return '\\newcommand{\\' + name + '}{' + val + '}'


mac = []
for n, r in R.items():
    t = tname(n)
    mac.append(nc('D' + t, f"{r['mean_dice']:.2f}"))
    mac.append(nc('J' + t, f"{r['mean_jaccard']:.2f}" if 'mean_jaccard' in r else '--'))
    mac.append(nc('T' + t, f"{r['seconds']:.0f}"))
    for k, i in enumerate(order):
        o = ['Liver', 'LK', 'RK', 'Spleen'][k]
        mac.append(nc('D' + t + o, f"{r['organ_dice'][i]:.2f}"))
        mac.append(nc('J' + t + o, f"{r['organ_jaccard'][i]:.2f}" if 'organ_jaccard' in r else '--'))
for s_ in stats:
    t = tname(s_[0])
    mac.append(nc('P' + t, s_[3]))
    mac.append(nc('Diff' + t, s_[2]))
for k, v in fa.items():
    mac.append(nc('AUC' + k, f"{v['auc']:.3f}"))
open(os.path.join(out, 'numbers.tex'), 'w').write('\n'.join(mac) + '\n')


# Qualitative figure: input / GT / prediction for 3 patients
pred_dir = os.path.join(src, 'pred')
runs = [('source_only', 'Source only'), ('BN_K0.1', 'ODES K=10'), ('FREQ_K0.1', 'FreqODES K=10')]
if all(os.path.exists(os.path.join(pred_dir, f'{n}_run_0.npz')) for n, _ in runs):
    from matplotlib.colors import ListedColormap
    P = {n: np.load(os.path.join(pred_dir, f'{n}_run_0.npz')) for n, _ in runs}
    ids = sorted({int(k.split('_')[1]) for k in P['source_only'].files if k.startswith('img_')})
    cmap = ListedColormap([(0, 0, 0, 0), (.9, .2, .2, .6), (.2, .8, .2, .6), (.2, .4, 1, .6), (1, .8, 0, .6)])
    def lab(x):  # class 4 = background -> 0, organs 0..3 -> 1..4
        return np.where(x == 4, 0, x + 1)
    fig, axs = plt.subplots(len(ids), 2 + len(runs), figsize=(2.3 * (2 + len(runs)), 2.4 * len(ids)))
    for r_, i in enumerate(ids):
        gt = P['source_only'][f'gt_{i}']
        z = int(np.argmax((gt != 4).sum((1, 2))))
        img = P['source_only'][f'img_{i}'][z].astype(np.float32)
        panels = [('Input (OOP MRI)', None), ('Ground truth', gt[z])] + [(l, P[n][f'pred_{i}'][z]) for n, l in runs]
        for c, (title, msk) in enumerate(panels):
            a = axs[r_, c]; a.imshow(img, cmap='gray')
            if msk is not None:
                a.imshow(lab(msk), cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            if r_ == 0: a.set_title(title, fontsize=9)
        axs[r_, 0].set_ylabel(f'Patient #{i + 1}, slice {z}', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'qualitative.png'), dpi=150)

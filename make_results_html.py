"""Build a self-contained results dashboard (FreqODES v1 vs ODES) from results/json.
    python make_results_html.py results ../FreqODES_v1_results.html
"""
import base64
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

src, out = sys.argv[1], sys.argv[2]
R = {os.path.basename(f).replace('_run_0.json', ''): json.load(open(f))
     for f in glob.glob(os.path.join(src, 'json', '*.json')) if not os.path.basename(f).startswith('V2_')}
FA = json.load(open(os.path.join(src, 'freq_analysis.json')))
organs = R['source_only']['organs']
ORDER = [organs.index(o) for o in ('liver', 'left kidney', 'right kidney', 'spleen')]
ORG = ['Liver', 'Left kidney', 'Right kidney', 'Spleen']


def org(n, key='organ_dice'):
    return [round(R[n][key][i], 2) for i in ORDER]


def wil(a, b):
    x, y = np.array(R[a]['per_batch_mean_dice']), np.array(R[b]['per_batch_mean_dice'])
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return round(float(np.mean(x - y)), 2), round(float(p), 3)


def img(name):
    p = os.path.join(src, 'figures', name)
    return 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode() if os.path.exists(p) else ''


METHODS = [  # key prefix, label, css var
    ('BN', 'ODES (BN-KL)', 'var(--s1)'),
    ('FFT', 'FreqODES FFT', 'var(--s2)'),
    ('DWT', 'FreqODES DWT', 'var(--s5)'),
    ('FREQ', 'FreqODES FFT+DWT', 'var(--s4)'),
    ('FUSED', 'Fused λ=0.5', 'var(--s3)'),
    ('RANDOM', 'Random', 'var(--s6)'),
]


def key(p, K):
    return f'{p}_K{K}' + ('_lam0.5' if p == 'FUSED' else '')


data = {
    'organs': ORG,
    'methods': [{'id': p, 'label': l, 'color': c,
                 'k10': R[key(p, '0.1')]['mean_dice'], 'k50': R[key(p, '0.5')]['mean_dice'],
                 'k100': R['ODES_K100']['mean_dice'],
                 'iou10': R[key(p, '0.1')]['mean_jaccard'], 'iou50': R[key(p, '0.5')]['mean_jaccard'],
                 't10': R[key(p, '0.1')]['seconds'], 't50': R[key(p, '0.5')]['seconds'],
                 'org10': org(key(p, '0.1')), 'org50': org(key(p, '0.5')),
                 'w10': wil(key(p, '0.1'), 'BN_K0.1'), 'w50': wil(key(p, '0.5'), 'BN_K0.5')}
                for p, l, c in METHODS],
    'paper': {'source': [87.77, 37.97, 18.92, 67.31, 52.99], 'k100': [88.90, 72.41, 74.55, 78.56, 78.61],
              'k50': [89.13, 72.30, 74.43, 78.36, 78.56], 'k10': [88.41, 71.45, 70.22, 77.12, 76.80]},
    'ours': {'source': org('source_only') + [R['source_only']['mean_dice']], 'k100': org('ODES_K100') + [R['ODES_K100']['mean_dice']],
             'k50': org('BN_K0.5') + [R['BN_K0.5']['mean_dice']], 'k10': org('BN_K0.1') + [R['BN_K0.1']['mean_dice']]},
    'oursIou': {k: R[n]['mean_jaccard'] for k, n in (('source', 'source_only'), ('k100', 'ODES_K100'), ('k50', 'BN_K0.5'), ('k10', 'BN_K0.1'))},
    'patients': {'BN': R['BN_K0.1']['per_batch_mean_dice'], 'FREQ': R['FREQ_K0.1']['per_batch_mean_dice'],
                 'FUSED25': R['FUSED_K0.1_lam0.25']['per_batch_mean_dice'], 'SRC': R['source_only']['per_batch_mean_dice'],
                 'ALL': R['ODES_K100']['per_batch_mean_dice']},
    'lam': [[0, R['FREQ_K0.1']['mean_dice']], [0.25, R['FUSED_K0.1_lam0.25']['mean_dice']], [0.5, R['FUSED_K0.1_lam0.5']['mean_dice']],
            [0.75, R['FUSED_K0.1_lam0.75']['mean_dice']], [1, R['BN_K0.1']['mean_dice']]],
    'dw': [['ODES (BN-KL)', R['BN_K0.1_noDW']['mean_dice'], R['BN_K0.1']['mean_dice']],
           ['FreqODES FFT+DWT', R['FREQ_K0.1_noDW']['mean_dice'], R['FREQ_K0.1']['mean_dice']],
           ['Random', R['RANDOM_K0.1_noDW']['mean_dice'], R['RANDOM_K0.1']['mean_dice']]],
    'auc': [[k, v['auc'], v['src_mean'], v['tgt_mean']] for k, v in FA.items()],
    'source': R['source_only']['mean_dice'], 'sourceIou': R['source_only']['mean_jaccard'],
    'fused25': R['FUSED_K0.1_lam0.25']['mean_dice'], 'fused25iou': R['FUSED_K0.1_lam0.25']['mean_jaccard'],
    'fused25t': R['FUSED_K0.1_lam0.25']['seconds'], 'fused25w': wil('FUSED_K0.1_lam0.25', 'BN_K0.1'),
    'v2': json.load(open(os.path.join(src, 'v2_summary.json'))) if os.path.exists(os.path.join(src, 'v2_summary.json')) else None,
}
IMG = {k: img(k + '.png') for k in ('freq_analysis', 'qualitative', 'train_loss', 'online_dice', 'v2_results')}

html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_template.html'), encoding='utf-8').read()
html = html.replace('/*__DATA__*/null', json.dumps(data))
for k, v in IMG.items():
    html = html.replace(f'__IMG_{k}__', v)
open(out, 'w', encoding='utf-8').write(html)
print('wrote', out, len(html) // 1024, 'KB')

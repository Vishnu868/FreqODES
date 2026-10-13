"""Diversity utilities used by methods/ODES.py.

The public ODES repository imports this module but does not ship it, so it is
re-implemented here from Sec. 2 / Algorithm 2 of the paper:
  * farthest_first_with_reference_distance -> image (slice) pruning that starts
    from the most domain-shifted slice and spreads the K% picks in feature space
  * get_weight_map -> diversity weight map W = Wd (spatial) * Wf (feature),
    built with Farthest-First Traversal (FFT) + Gaussian kernels.
("FFT" in this file means Farthest-First Traversal, not Fourier transform.)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _farthest_first(dist_matrix, n_select, start_idx):
    """Algorithm 2: greedy farthest-first traversal on a precomputed distance matrix."""
    n = dist_matrix.shape[0]
    n_select = min(n_select, n)
    selected = [start_idx]
    min_dist = dist_matrix[start_idx].copy()
    for _ in range(1, n_select):
        min_dist[selected] = -1
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist_matrix[nxt])
    return np.array(selected)


def farthest_first_with_reference_distance(features, dists_from_source, k_percent, device):
    """Select ceil(K% * B) slices of a batch.

    The first pick is the slice with the largest divergence from the source.
    Every next pick maximises (min feature distance to the picked set) x (its
    divergence from the source), so the picks are both shifted and non-redundant.
    """
    b = features.shape[0]
    n_select = max(1, int(math.ceil(b * k_percent / 100.0)))
    if n_select >= b:
        return np.arange(b)

    feat = F.adaptive_avg_pool2d(features.float(), 1).flatten(1)
    feat = F.normalize(feat, dim=1)
    feat_dist = (1 - feat @ feat.t()).clamp(min=0).cpu().numpy()

    ref = dists_from_source.float().cpu().numpy()
    ref = ref / (ref.max() + 1e-12)

    selected = [int(np.argmax(ref))]
    min_dist = feat_dist[selected[0]].copy()
    for _ in range(1, n_select):
        score = min_dist * ref
        score[selected] = -1
        nxt = int(np.argmax(score))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, feat_dist[nxt])
    return np.array(sorted(selected))


def _gaussian_sum(points, shape, sigma):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros(shape, dtype=np.float32)
    for (py, px) in points:
        out += np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2.0 * sigma ** 2))
    return out / (out.max() + 1e-12)


def get_weight_map(heat_map, threshold, features, k_percent, alpha_cos=5, alpha_eu=5, plot=0):
    """Diversity weight map of the paper (Fig. 3).

    heat_map : (H, W) numpy acquisition map A_init, normalised to [0, 1]
    features : (C, h, w) feature map of the same image
    Returns (weight (h, w) numpy, cosine distance matrix, euclidean distance matrix).
    A map of ones is returned when there are no high-uncertainty candidates.
    """
    c, h, w = features.shape
    hm = torch.as_tensor(np.ascontiguousarray(heat_map), dtype=torch.float32)[None, None]
    hm = F.adaptive_max_pool2d(hm, (h, w))[0, 0].numpy()

    cand = np.argwhere(hm > threshold * hm.max()) if hm.max() > 0 else np.zeros((0, 2))
    n_select = max(1, int(round(h * w * k_percent / 100.0)))
    if len(cand) <= 1:
        return np.ones((h, w), dtype=np.float32), None, None

    coords = cand.astype(np.float32)
    euclidean_distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    f = features[:, cand[:, 0], cand[:, 1]].t().float()
    f = F.normalize(f, dim=1)
    cosine_distance = (1 - f @ f.t()).clamp(min=0).cpu().numpy()

    start = int(np.argmax(hm[cand[:, 0], cand[:, 1]]))
    sel_eu = _farthest_first(euclidean_distance, n_select, start)
    sel_cos = _farthest_first(cosine_distance, n_select, start)

    sigma_eu = max(h, w) / (2.0 * alpha_eu) * 2
    sigma_cos = max(h, w) / (2.0 * alpha_cos) * 2
    w_eu = _gaussian_sum(cand[sel_eu], (h, w), sigma_eu)
    w_cos = _gaussian_sum(cand[sel_cos], (h, w), sigma_cos)

    weight = w_eu * w_cos
    weight = weight / (weight.max() + 1e-12)
    return weight.astype(np.float32), cosine_distance, euclidean_distance


def expand_labels_with_cosine_similarity(label, similarity, threshold):
    """Unused by the released pipeline (call is commented out); kept for import compatibility."""
    return label

"""FreqODES novelty: frequency-domain domain-shift score for image pruning.

ODES ranks the slices of an incoming batch by the KL divergence between
BN statistics of the source model and those of the (augmented) slice.
Here the same ranking is computed directly from the image:

  * FFT descriptor: fraction of spectral energy in N_BANDS concentric radial
    frequency bands of the centred 2-D FFT magnitude spectrum (log-scaled).
  * DWT descriptor: relative energy of the LL / LH / HL / HH sub-bands of a
    LEVELS-level Haar wavelet decomposition (log-scaled).

A reference distribution (per-feature mean / std) is estimated once from a
small set of source-domain slices; the shift score of a target slice is the
mean squared z-score of its descriptor (a diagonal Mahalanobis distance).
No network forward pass is needed, so the score is cheap and interpretable.
"""
import numpy as np
import pywt

N_BANDS = 16
LEVELS = 3
EPS = 1e-8


def _radial_index(h, w, n_bands):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    r = r / r.max()
    return np.minimum((r * n_bands).astype(int), n_bands - 1)


_RADIAL_CACHE = {}


def fft_descriptor(img, n_bands=N_BANDS):
    """img: (H, W) float array. Returns log fraction of energy per radial band."""
    h, w = img.shape
    key = (h, w, n_bands)
    if key not in _RADIAL_CACHE:
        _RADIAL_CACHE[key] = _radial_index(h, w, n_bands)
    idx = _RADIAL_CACHE[key]
    x = img - img.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    band = np.bincount(idx.ravel(), weights=power.ravel(), minlength=n_bands)
    band = band / (band.sum() + EPS)
    return np.log(band + EPS)


def dwt_descriptor(img, levels=LEVELS, wavelet='haar'):
    """Log relative energy of every DWT sub-band (LL_L, then LH/HL/HH per level)."""
    coeffs = pywt.wavedec2(img - img.mean(), wavelet, level=levels)
    energies = [np.mean(coeffs[0] ** 2)]
    for (lh, hl, hh) in coeffs[1:]:
        energies += [np.mean(lh ** 2), np.mean(hl ** 2), np.mean(hh ** 2)]
    e = np.array(energies)
    e = e / (e.sum() + EPS)
    return np.log(e + EPS)


DESCRIPTORS = {'FFT': fft_descriptor, 'DWT': dwt_descriptor}


class FrequencyReference:
    """Source-domain reference statistics for the FFT and DWT descriptors."""

    def __init__(self, source_slices):
        """source_slices: iterable of (H, W) numpy arrays from the source domain."""
        self.stats = {}
        feats = {k: [] for k in DESCRIPTORS}
        for s in source_slices:
            for k, fn in DESCRIPTORS.items():
                feats[k].append(fn(s))
        for k, v in feats.items():
            v = np.stack(v)
            self.stats[k] = (v.mean(0), v.std(0) + 1e-3)
        self.n_ref = len(feats['FFT'])

    def score(self, img, kind):
        """Shift score of one (H, W) slice. kind in {'FFT', 'DWT', 'FREQ'}."""
        if kind == 'FREQ':
            return 0.5 * (self.score(img, 'FFT') + self.score(img, 'DWT'))
        mu, sd = self.stats[kind]
        z = (DESCRIPTORS[kind](img) - mu) / sd
        return float(np.mean(z ** 2))

    def score_stack(self, stack, kind):
        """stack: (B, H, W) numpy. Returns (B,) scores."""
        return np.array([self.score(s, kind) for s in stack])

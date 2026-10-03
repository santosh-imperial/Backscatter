"""Fixed Gabor bank: sampled BSE image texture, never phase or plate identity."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.signal import fftconvolve

KEYS = ('gabor_coarse_energy_share', 'gabor_axial_strength',
        'gabor_horizontal_wavevector_balance')
PERIODS = (16, 32, 64)
ANGLES = np.arange(4) * np.pi / 4
WINDOW = 512
GUARD_RAW = 96
MODES = {'nominal': (2, 1., False), 'full_resolution': (1, 1., False),
         'gamma_075': (2, .75, False), 'gamma_125': (2, 1.25, False),
         'affine': (2, 1., True)}
DEFINITIONS = {
    KEYS[0]: ('fraction', 'Sum of mean squared complex filter responses at period 64 px / sum across periods 16,32,64 px and four wavevector directions, pooled equally across four fixed windows.'),
    KEYS[1]: ('0–1', 'Absolute energy-weighted mean exp(2j theta) across four wavevector directions, all three periods and four fixed windows. Direction is the modulation wavevector, normal to a stripe.'),
    KEYS[2]: ('−1–1', 'Energy-weighted mean cos(2 theta) across all periods/windows; positive means horizontal wavevectors (vertical stripes), not horizontal graphite plates.'),
}


def windows(shape):
    """Four non-overlapping 512px windows centred at quarter-frame coordinates."""
    h, w = shape
    if min(h, w) < 2*WINDOW:
        return []
    return [(int(h*f)-WINDOW//2, int(w*g)-WINDOW//2)
            for f in (.25, .75) for g in (.25, .75)]


@lru_cache(maxsize=None)
def kernel(period_raw, theta, step):
    period = period_raw / step
    sigma = period / 2
    radius = int(np.ceil(3*sigma))
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    k = np.exp(-.5*(x*x+y*y)/sigma**2 +
               2j*np.pi*(x*np.cos(theta)+y*np.sin(theta))/period)
    k -= k.mean()  # exact finite-support DC removal
    k /= np.sqrt(np.sum(abs(k)**2))
    return k


def summary(energy):
    energy = np.asarray(energy, float)
    unavailable = {k: np.nan for k in KEYS}
    if energy.shape != (3, 4) or not np.isfinite(energy).all() or (energy < 0).any():
        return unavailable
    total = energy.sum()
    if total <= 1e-12:
        return unavailable
    directional = energy.sum(axis=0)
    moment = np.sum(directional*np.exp(2j*ANGLES))/total
    return {KEYS[0]: float(energy[2].sum()/total),
            KEYS[1]: float(abs(moment)), KEYS[2]: float(moment.real)}


def measure_window(raw, mode='nominal', keep_maps=False):
    """Mean-zero/unit-SD window; only the common valid convolution interior."""
    if mode not in MODES:
        raise ValueError('Unknown fixed mode')
    raw = np.asarray(raw, float)
    if raw.shape != (WINDOW, WINDOW) or not np.isfinite(raw).all():
        return None, {}, 'invalid_window'
    if raw.min() < 0 or raw.max() > 255:
        return None, {}, 'outside_8bit_range'
    step, gamma, affine = MODES[mode]
    a = (raw/255)**gamma
    if affine:
        a = .1 + .7*a  # no clipping
    if step > 1:
        a = a.reshape(WINDOW//step, step, WINDOW//step, step).mean(axis=(1,3))
    sd = a.std()
    if sd <= 1e-6:
        return None, {}, 'insufficient_intensity_variation'
    a = (a-a.mean())/sd
    guard = GUARD_RAW//step
    valid = np.s_[guard:-guard, guard:-guard]
    energy = np.zeros((3, 4))
    maps = {}
    for si, period in enumerate(PERIODS):
        for oi, theta in enumerate(ANGLES):
            response = fftconvolve(a, kernel(period, float(theta), step), mode='same')
            power = abs(response)**2
            energy[si, oi] = power[valid].mean()
            if keep_maps:
                maps[(period, oi)] = power[valid]
    return energy, maps, ''


def usable(table, key, view='all_known'):
    good = np.isfinite(table[key]) & table.gabor_windows_used.eq(4)
    if view in ('quality_matched', 'ordinary_reference'):
        for flag in ('bright_low_contrast', 'grey_pore'):
            good &= table[flag].eq(False).fillna(False) if flag in table else False
    if view == 'ordinary_reference':
        good &= ~table.batch.eq('Batch_3') | table.cracked_known.eq(False).fillna(False)
    return np.asarray(good, bool)

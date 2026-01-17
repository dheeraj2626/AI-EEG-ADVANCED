import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}

def compute_band_powers(signal, fs=250):
    """
    signal: numpy array (channels, time)
    returns dict of band powers
    """
    band_powers = {band: [] for band in BANDS}

    for ch_signal in signal:
        nperseg = min(len(ch_signal), fs * 2)
        freqs, psd = welch(ch_signal, fs=fs, nperseg=nperseg)

        for band, (low, high) in BANDS.items():
            idx = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[idx])
            band_powers[band].append(band_power)

    # Average across channels
    return {band: float(np.mean(vals)) for band, vals in band_powers.items()}

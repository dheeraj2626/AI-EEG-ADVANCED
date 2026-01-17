# quality_analysis/artifact_classifier.py

import numpy as np
from scipy.signal import welch

def band_power(signal, fs, band):
    nperseg = min(len(signal), fs*2)
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.sum(psd[idx])


def classify_artifact(signal, fs=250):
    """
    signal: 1D EEG segment
    returns artifact type string
    """

    delta = band_power(signal, fs, (0.5, 4))
    theta = band_power(signal, fs, (4, 8))
    alpha = band_power(signal, fs, (8, 13))
    beta = band_power(signal, fs, (13, 30))
    gamma = band_power(signal, fs, (30, 80))

    total = delta + theta + alpha + beta + gamma + 1e-6

    # Relative powers
    delta_r = delta / total
    gamma_r = gamma / total

    # Powerline check (50 Hz)
    powerline = band_power(signal, fs, (48, 52))

    if delta_r > 0.45:
        return "Eye Blink (EOG)"
    elif gamma_r > 0.35:
        return "Muscle (EMG)"
    elif powerline > 0.2 * total:
        return "Powerline Noise"
    elif np.std(signal) > 4 * np.median(np.abs(signal)):
        return "Motion Artifact"
    else:
        return "Unknown / Mixed"

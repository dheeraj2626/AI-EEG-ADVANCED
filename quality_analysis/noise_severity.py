# quality_analysis/noise_severity.py

import numpy as np
from .snr import calculate_snr
from scipy.stats import entropy

def compute_noise_severity(noisy_signal, clean_signal):
    snr = calculate_snr(clean_signal, noisy_signal)

    variance = np.var(noisy_signal)
    hist, _ = np.histogram(noisy_signal, bins=100, density=True)
    ent = entropy(hist + 1e-10)

    # Normalize components
    snr_score = np.clip((snr + 20) / 80, 0, 1)
    var_score = np.clip(variance / (variance + 1), 0, 1)
    ent_score = np.clip(ent / (ent + 1), 0, 1)

    severity = (0.5 * (1 - snr_score) +
                0.3 * var_score +
                0.2 * ent_score)

    return round(severity * 100, 2)

def quality_label(severity_score):
    if severity_score < 30:
        return "Good"
    elif severity_score < 60:
        return "Fair"
    else:
        return "Poor"

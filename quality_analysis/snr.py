# quality_analysis/snr.py

import numpy as np

def calculate_snr(clean_signal, noisy_signal):
    """
    SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noisy_signal - clean_signal) ** 2)

    if noise_power == 0:
        return 100.0

    snr = 10 * np.log10(signal_power / noise_power)
    return float(np.clip(snr, -20, 60))

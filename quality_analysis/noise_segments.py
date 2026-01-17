import numpy as np

def robust_std(signal):
    """
    Robust standard deviation using MAD
    """
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return 1.4826 * mad + 1e-6  # avoid zero


def detect_noisy_segments(
    eeg_signal,
    window_size=200,
    step_size=100,
    threshold_factor=3.0
):
    """
    eeg_signal: numpy array (batch, channels, time)
    returns list of noisy segments
    """

    batch, channels, time_len = eeg_signal.shape
    noisy_segments = []

    for b in range(batch):
        for ch in range(channels):
            signal = eeg_signal[b, ch]

            baseline_std = robust_std(signal)

            for start in range(0, time_len - window_size, step_size):
                end = start + window_size
                window = signal[start:end]

                window_std = np.std(window)

                if window_std > threshold_factor * baseline_std:
                    noisy_segments.append({
                        "batch": b,
                        "channel": ch,
                        "start": start,
                        "end": end,
                        "severity": round(window_std / baseline_std, 2)
                    })

    return noisy_segments


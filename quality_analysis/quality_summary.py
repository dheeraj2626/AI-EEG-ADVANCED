from collections import Counter
import numpy as np


def summarize_signal_quality(
    noisy_segments,
    total_timepoints,
    fs=250
):
    """
    noisy_segments: list of dicts from detect_noisy_segments()
    total_timepoints: total EEG samples (time dimension)
    """

    if len(noisy_segments) == 0:
        return {
            "quality": "Good",
            "noise_percentage": 0.0,
            "avg_severity": 0.0,
            "dominant_artifact": "None",
            "recommendation": "EEG signal quality is good. Suitable for analysis."
        }

    # ---- 1. Noise duration ----
    noisy_samples = set()
    severities = []
    artifact_labels = []

    for seg in noisy_segments:
        noisy_samples.update(range(seg["start"], seg["end"]))
        severities.append(seg.get("severity", 1.0))
        if "artifact" in seg:
            artifact_labels.append(seg["artifact"])

    noise_percentage = (len(noisy_samples) / total_timepoints) * 100
    avg_severity = round(float(np.mean(severities)), 2)

    # ---- 2. Dominant artifact ----
    if artifact_labels:
        dominant_artifact = Counter(artifact_labels).most_common(1)[0][0]
    else:
        dominant_artifact = "Unknown"

    # ---- 3. Quality label ----
    if noise_percentage < 10:
        quality = "Good"
    elif noise_percentage < 30:
        quality = "Fair"
    else:
        quality = "Poor"

    # ---- 4. Recommendation ----
    if quality == "Good":
        recommendation = "EEG signal quality is good. Suitable for reliable analysis."
    elif quality == "Fair":
        recommendation = (
            "EEG signal shows moderate noise. Interpret results with caution."
        )
    else:
        recommendation = (
            "EEG signal quality is poor. Re-recording or advanced cleaning recommended."
        )

    return {
        "quality": quality,
        "noise_percentage": round(noise_percentage, 2),
        "avg_severity": avg_severity,
        "dominant_artifact": dominant_artifact,
        "recommendation": recommendation
    }

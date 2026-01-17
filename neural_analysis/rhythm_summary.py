def summarize_neural_rhythms(band_powers):
    """
    band_powers: dict from compute_band_powers
    returns interpreted neural patterns
    """
    total_power = sum(band_powers.values()) + 1e-6
    relative = {k: v / total_power for k, v in band_powers.items()}

    dominant_band = max(relative, key=relative.get)

    interpretations = []

    if relative["beta"] > 0.35:
        interpretations.append(
            "Elevated beta activity suggests increased cognitive arousal or mental workload."
        )

    if relative["alpha"] < 0.15:
        interpretations.append(
            "Reduced alpha activity may indicate difficulty achieving a relaxed state."
        )

    if relative["theta"] > 0.30:
        interpretations.append(
            "Increased theta activity is commonly associated with drowsiness or internal focus."
        )

    if not interpretations:
        interpretations.append(
            "Neural rhythm distribution appears within typical ranges."
        )

    return {
        "dominant_rhythm": dominant_band,
        "relative_power": {k: round(v, 3) for k, v in relative.items()},
        "interpretation": interpretations
    }

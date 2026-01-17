def generate_json_report(
    signal_quality_summary,
    neural_rhythm_summary
):
    """
    Generates a structured JSON-style EEG report.
    """

    return {
        "signal_quality": signal_quality_summary,
        "neural_activity": {
            "dominant_rhythm": neural_rhythm_summary["dominant_rhythm"],
            "relative_band_powers": neural_rhythm_summary["relative_power"],
        },
        "interpretation": neural_rhythm_summary["interpretation"],
        "recommendation": signal_quality_summary["recommendation"],
    }

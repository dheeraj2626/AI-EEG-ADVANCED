def generate_text_report(
    signal_quality_summary,
    neural_rhythm_summary
):
    """
    Generates a human-readable EEG neural analysis report.
    """

    lines = []

    # ---- Signal Quality ----
    lines.append("EEG SIGNAL QUALITY SUMMARY")
    lines.append("-" * 30)
    lines.append(f"Overall Quality       : {signal_quality_summary['quality']}")
    lines.append(f"Noise Percentage (%)  : {signal_quality_summary['noise_percentage']}")
    lines.append(f"Dominant Artifact     : {signal_quality_summary['dominant_artifact']}")
    lines.append(f"Average Severity      : {signal_quality_summary['avg_severity']}")
    lines.append("")

    # ---- Neural Activity ----
    lines.append("NEURAL RHYTHM ANALYSIS")
    lines.append("-" * 30)
    lines.append(f"Dominant Rhythm       : {neural_rhythm_summary['dominant_rhythm']}")
    lines.append("Relative Band Powers:")
    for band, val in neural_rhythm_summary["relative_power"].items():
        lines.append(f"  {band.capitalize():<8}: {val}")

    lines.append("")

    # ---- Interpretation ----
    lines.append("INTERPRETATION")
    lines.append("-" * 30)
    for line in neural_rhythm_summary["interpretation"]:
        lines.append(f"- {line}")

    lines.append("")

    # ---- Recommendation ----
    lines.append("RECOMMENDATION")
    lines.append("-" * 30)
    lines.append(signal_quality_summary["recommendation"])

    return "\n".join(lines)

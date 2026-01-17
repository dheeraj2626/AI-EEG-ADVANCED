import sys
import os

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core_denoiser.denoise import denoise_eeg
from quality_analysis.noise_segments import detect_noisy_segments
from quality_analysis.artifact_classifier import classify_artifact
from quality_analysis.quality_summary import summarize_signal_quality

from neural_analysis.band_power import compute_band_powers
from neural_analysis.rhythm_summary import summarize_neural_rhythms

from report_generator.report_text import generate_text_report
from report_generator.report_json import generate_json_report
from report_generator.report_pdf import generate_pdf_report


# ---------------- UI SETUP ----------------
st.set_page_config(page_title="AI-EEG Advanced", layout="wide")
st.title("🧠 AI-EEG Advanced Neural Analysis System")

st.sidebar.header("EEG Input")
use_demo = st.sidebar.checkbox("Use demo EEG data")

# ---------------- LOAD EEG ----------------
if use_demo:
    eeg = np.random.randn(1, 19, 1000).astype("float32")
    eeg[0, 3, 400:550] += 10.0
    eeg[0, 7, 200:300] += 8.0
else:
    st.info("Enable demo EEG data to proceed.")
    st.stop()

# ---------------- DENOISING ----------------
with st.spinner("Denoising EEG..."):
    clean = denoise_eeg(eeg)

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Raw vs Denoised EEG (Channel 0)")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(eeg[0, 0], label="Raw EEG", alpha=0.6)
ax.plot(clean[0, 0], label="Denoised EEG", linewidth=2)
ax.legend()
st.pyplot(fig)

# ---------------- NOISE ANALYSIS ----------------
segments = detect_noisy_segments(eeg)

for seg in segments:
    ch = seg["channel"]
    seg_signal = eeg[0, ch, seg["start"]:seg["end"]]
    seg["artifact"] = classify_artifact(seg_signal)

summary = summarize_signal_quality(
    noisy_segments=segments,
    total_timepoints=eeg.shape[2]
)

# ---------------- SIGNAL QUALITY ----------------
st.subheader("📊 Signal Quality Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Quality", summary["quality"])
c2.metric("Noise %", summary["noise_percentage"])
c3.metric("Avg Severity", summary["avg_severity"])
c4.metric("Artifact", summary["dominant_artifact"])
st.warning(summary["recommendation"])

# ---------------- NEURAL ANALYSIS ----------------
band_powers = compute_band_powers(clean[0])
neural_summary = summarize_neural_rhythms(band_powers)

st.subheader("🧠 Neural Rhythm Analysis")
st.write("**Dominant Rhythm:**", neural_summary["dominant_rhythm"])
st.json(neural_summary["relative_power"])

for line in neural_summary["interpretation"]:
    st.write("•", line)

# ---------------- FINAL REPORT ----------------
st.subheader("📄 Neural Analysis Report")

text_report = generate_text_report(summary, neural_summary)
st.text_area("Generated Report", text_report, height=350)

# ---------------- PDF DOWNLOAD ----------------
pdf_buffer = generate_pdf_report(text_report)

st.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer,
    file_name="AI_EEG_Neural_Report.pdf",
    mime="application/pdf",
)

st.success("Report generated successfully!")





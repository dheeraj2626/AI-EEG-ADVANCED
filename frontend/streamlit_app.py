import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # ✅ NEW

from core_denoiser.denoise import denoise_eeg
from neural_analysis.band_power import compute_band_powers
from neural_analysis.rhythm_summary import summarize_neural_rhythms
from report_generator.report_pdf import generate_pdf_report


# ===================== HELPERS =====================

def compute_noise_metrics(signal):
    """
    Computes noise metrics safely.
    Ensures compatibility with compute_band_powers().
    """

    signal = np.asarray(signal).reshape(-1)

    band_powers = compute_band_powers(signal.reshape(1, -1))

    hf_power = band_powers.get("beta", 0.0) + band_powers.get("gamma", 0.0)
    total_power = sum(band_powers.values()) + 1e-8

    noise_percentage = (hf_power / total_power) * 100.0

    severity = float(
        np.std(signal) + np.mean(np.abs(np.diff(signal)))
    )

    return noise_percentage, severity, band_powers


def infer_mental_state(band_powers):
    dominant = max(band_powers, key=band_powers.get)

    states = {
        "delta": "Deep sleep / unconscious",
        "theta": "Drowsy / meditative",
        "alpha": "Relaxed / calm",
        "beta": "Focused / alert",
        "gamma": "High cognitive load"
    }
    return dominant, states.get(dominant, "Mixed state")


def generate_mental_report(band_powers):
    """
    Generates a detailed, human-readable mental state analysis
    based on EEG band powers.
    """

    total = sum(band_powers.values()) + 1e-8
    rel = {k: v / total for k, v in band_powers.items()}

    alpha = rel.get("alpha", 0)
    beta = rel.get("beta", 0)
    gamma = rel.get("gamma", 0)
    theta = rel.get("theta", 0)
    delta = rel.get("delta", 0)

    # Interpretations
    relaxation = "High" if alpha > 0.3 else "Moderate" if alpha > 0.15 else "Low"
    focus = "High" if beta > 0.25 else "Moderate" if beta > 0.15 else "Low"
    cognitive_load = "High" if gamma > 0.15 else "Moderate" if gamma > 0.07 else "Low"
    fatigue = "High" if theta > 0.25 else "Moderate" if theta > 0.15 else "Low"

    thought_stability = (
        "Stable" if abs(alpha - beta) < 0.1
        else "Rapidly shifting"
    )

    balance = "Well balanced" if max(rel.values()) < 0.4 else "Dominated by a single rhythm"

    report = f"""
Mental & Cognitive State Interpretation
---------------------------------------
• Relaxation Level      : {relaxation}
• Focus / Alertness     : {focus}
• Cognitive Load        : {cognitive_load}
• Mental Fatigue        : {fatigue}
• Thought Stability     : {thought_stability}
• Cognitive Balance     : {balance}

Note:
These observations describe the subject's *current mental state*
and are not intended for medical diagnosis.
"""

    return report

def compute_denoising_confidence(
    raw_noise_pct,
    clean_noise_pct,
    raw_severity,
    clean_severity,
    raw_band_powers,
    clean_band_powers
):
    """
    Computes a denoising confidence score (0–100)
    based on noise reduction, severity reduction,
    and spectral stabilization.
    """

    # ---- Noise reduction score ----
    noise_improvement = max(raw_noise_pct - clean_noise_pct, 0.0)
    noise_score = min(noise_improvement * 2, 100)  # scale

    # ---- Severity reduction score ----
    severity_improvement = max(raw_severity - clean_severity, 0.0)
    severity_score = min(severity_improvement * 15, 100)

    # ---- Spectral stability score ----
    def spectral_variance(bands):
        vals = np.array(list(bands.values()))
        return np.std(vals)

    raw_spec_var = spectral_variance(raw_band_powers)
    clean_spec_var = spectral_variance(clean_band_powers)

    spectral_improvement = max(raw_spec_var - clean_spec_var, 0.0)
    spectral_score = min(spectral_improvement * 50, 100)

    # ---- Final confidence ----
    confidence = (
        0.5 * noise_score +
        0.3 * severity_score +
        0.2 * spectral_score
    )

    confidence = confidence * 5  # scale factor
    return round(min(confidence, 100), 2)



def cleaned_eeg_to_csv(clean_eeg, channel_names):
    """
    Converts full cleaned EEG to a CSV-friendly DataFrame.
    """
    rows = []
    for w in range(clean_eeg.shape[0]):
        for ch in range(clean_eeg.shape[1]):
            for s, val in enumerate(clean_eeg[w, ch]):
                rows.append([w, channel_names[ch], s, float(val)])

    return pd.DataFrame(
        rows,
        columns=["window", "channel", "sample", "amplitude"]
    )


# ===================== UI =====================
st.set_page_config(page_title="AI-EEG Advanced", layout="wide")
st.title("🧠 AI-EEG Advanced Neural Analysis System")

st.sidebar.header("EEG Input")
use_demo = st.sidebar.checkbox("Use demo EEG data")
uploaded_file = st.sidebar.file_uploader("Upload EEG file (.npy)", type=["npy"])


# ===================== LOAD EEG =====================
if use_demo:
    raw_eeg = np.random.randn(60, 19, 1000).astype("float32")
    raw_eeg += np.random.normal(0, 3.0, raw_eeg.shape)
    raw_eeg[10, 3, 400:550] += 15
    raw_eeg[25, 7, 200:300] += 12
    st.success("Using demo EEG data (noisy)")
elif uploaded_file is not None:
    raw_eeg = np.load(uploaded_file).astype("float32")
    if raw_eeg.ndim != 3:
        st.error("Expected EEG shape: (windows, channels, samples)")
        st.stop()
else:
    st.stop()

if "raw_eeg" not in st.session_state:
    st.session_state.raw_eeg = raw_eeg
    st.session_state.processed = False


# ===================== CONTROLS =====================
window_idx = st.sidebar.slider("EEG Window", 0, raw_eeg.shape[0] - 1, 0)

channel_names = [
    "Fp1","Fp2","F3","F4","C3","C4","P3","P4",
    "O1","O2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"
]

channel_idx = st.sidebar.selectbox(
    "EEG Channel",
    range(raw_eeg.shape[1]),
    format_func=lambda x: channel_names[x]
)

# ===================== RAW ANALYSIS =====================
raw_signal = st.session_state.raw_eeg[window_idx, channel_idx]

raw_noise_pct, raw_severity, raw_bands = compute_noise_metrics(raw_signal)
raw_rhythm, raw_state = infer_mental_state(raw_bands)

st.subheader("📈 Noisy EEG (Before Processing)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(raw_signal, color="blue", label="Noisy EEG")
ax.legend()
st.pyplot(fig)

st.subheader("📊 Before Processing")
st.write(f"**Noise %:** {raw_noise_pct:.2f}")
st.write(f"**Noise Severity:** {raw_severity:.2f}")
st.write(f"**Mental State:** {raw_state}")


# ===================== PROCESS =====================
if st.button("🚀 Process Data (Denoise & Compare)"):
    with st.spinner("Applying AI Denoising..."):
        clean_eeg = denoise_eeg(
            st.session_state.raw_eeg.reshape(
                1, st.session_state.raw_eeg.shape[1], -1
            )
        ).reshape(st.session_state.raw_eeg.shape)

    st.session_state.clean_eeg = clean_eeg
    st.session_state.processed = True


# ===================== AFTER PROCESSING =====================
# ===================== AFTER PROCESSING =====================
if st.session_state.processed:
    clean_signal = st.session_state.clean_eeg[window_idx, channel_idx]

    # ✅ FIRST compute clean metrics
    clean_noise_pct, clean_severity, clean_bands = compute_noise_metrics(clean_signal)
    clean_rhythm, clean_state = infer_mental_state(clean_bands)

    # ✅ THEN compute reductions
    noise_reduction = max(raw_noise_pct - clean_noise_pct, 0.0)
    severity_reduction = raw_severity - clean_severity

    # ✅ NOW compute denoising confidence (all vars exist)
    denoising_confidence = compute_denoising_confidence(
        raw_noise_pct,
        clean_noise_pct,
        raw_severity,
        clean_severity,
        raw_bands,
        clean_bands
    )

    # ---------- COMPARISON PLOT ----------
    st.subheader("📈 Noisy vs Processed EEG")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw_signal, color="blue", alpha=0.7, label="Noisy EEG")
    ax.plot(clean_signal, color="orange", linewidth=2, label="Processed EEG")
    ax.legend()
    st.pyplot(fig)

    # ---------- METRICS ----------
    st.subheader("📊 Denoising Impact")
    st.write(f"**Noise Reduced (%):** {noise_reduction:.2f}")
    st.write(f"**Severity Reduced:** {severity_reduction:.2f}")
    st.write(f"**Mental State After Processing:** {clean_state}")
    st.write(f"**Denoising Confidence:** {denoising_confidence:.2f}%")



    # ---------- CSV EXPORT (CURRENT WINDOW & CHANNEL) ----------
    df_current = pd.DataFrame({
        "window": window_idx,
        "channel": channel_names[channel_idx],
        "sample": np.arange(len(clean_signal)),
        "amplitude": clean_signal
    })

    st.download_button(
        label="⬇️ Download Cleaned EEG (Current Window & Channel CSV)",
        data=df_current.to_csv(index=False),
        file_name=f"cleaned_eeg_window{window_idx}_{channel_names[channel_idx]}.csv",
        mime="text/csv"
    )

    # ---------- CSV EXPORT (FULL EEG – OPTIONAL) ----------
    df_full = cleaned_eeg_to_csv(
        st.session_state.clean_eeg,
        channel_names
    )

    st.download_button(
        label="⬇️ Download FULL Cleaned EEG (CSV)",
        data=df_full.to_csv(index=False),
        file_name="cleaned_eeg_full.csv",
        mime="text/csv"
    )

    # ---------- PDF REPORT ----------
    mental_before = generate_mental_report(raw_bands)
    mental_after = generate_mental_report(clean_bands)
    
    report = f"""
    EEG DENOISING COMPARISON REPORT
    ==============================
    
    BEFORE PROCESSING
    -----------------
    Noise Percentage : {raw_noise_pct:.2f}
    Noise Severity   : {raw_severity:.2f}
    Mental State     : {raw_state}
    
    {mental_before}
    
    AFTER PROCESSING
    ----------------
    Noise Percentage : {clean_noise_pct:.2f}
    Noise Severity   : {clean_severity:.2f}
    Mental State     : {clean_state}
    
    {mental_after}
    
    IMPROVEMENT
    -----------
    Noise Reduced    : {noise_reduction:.2f}
    Severity Reduced : {severity_reduction:.2f}
    
    Observation:
    Improved signal quality leads to clearer and more stable
    mental state interpretation after denoising.
    
    IMPROVEMENT
    -----------
    Noise Reduced        : {noise_reduction:.2f}
    Severity Reduced     : {severity_reduction:.2f}
    Denoising Confidence : {denoising_confidence:.2f} %
    """


    pdf = generate_pdf_report(report)

    st.download_button(
        "📄 Download Comparison PDF",
        pdf,
        "AI_EEG_Noise_Comparison_Report.pdf",
        "application/pdf"
    )

    st.success("Processing complete, CSV & PDF exported successfully!")

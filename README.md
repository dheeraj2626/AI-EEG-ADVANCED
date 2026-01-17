📌 Project Overview

Electroencephalogram (EEG) signals are highly sensitive to noise and artifacts, which significantly affect brain signal analysis.
This project presents an AI-driven EEG preprocessing and analysis system that:

Automatically denoises EEG signals using a deep learning model

Quantifies noise and signal quality before and after processing

Analyzes neural rhythms (Delta, Theta, Alpha, Beta, Gamma)

Interprets mental and cognitive states in an explainable manner

Generates professional comparison reports and exports clean EEG data

The system is designed for educational, research, and experimental purposes.

🚀 Features

AI-based EEG denoising (UNet model)

Window-wise and channel-wise EEG analysis

Noise percentage and severity computation

EEG rhythm analysis (Δ, Θ, Α, Β, Γ)

Mental and cognitive state interpretation

Denoising confidence score (0–100%)

Before vs After EEG visualization

Automatic PDF report generation

CSV export of cleaned EEG data

Interactive Streamlit web interface

🧠 Mental State Interpretation

Based on EEG rhythm distribution, the system provides insights into:

Relaxation level

Focus and alertness

Cognitive load

Mental fatigue

Thought stability

Overall cognitive balance

⚠️ These interpretations describe temporary mental states and are not medical diagnoses.

🛠️ Tech Stack

Programming Language: Python

Frontend & UI: Streamlit

Deep Learning: PyTorch

Signal Processing: NumPy, SciPy

Visualization: Matplotlib

Data Handling: Pandas

Report Generation: ReportLab

📂 Project Structure
AI-EEG-Advanced/
│
├── frontend/
│   └── streamlit_app.py
│
├── core_denoiser/
│   └── denoise.py
│
├── neural_analysis/
│   ├── band_power.py
│   └── rhythm_summary.py
│
├── report_generator/
│   └── report_pdf.py
│
├── requirements.txt
├── README.md
└── .gitignore

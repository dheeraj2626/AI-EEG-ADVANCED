import torch
import os
import numpy as np
from core_denoiser.unet1d import UNet1D

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "unet_eeg_denoiser.pt"
)

device = torch.device("cpu")

# Model trained on 19 EEG channels
model = UNet1D(channels=19)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def denoise_eeg(eeg_signal):
    """
    eeg_signal: numpy array of shape (batch, 19, time)
    returns: numpy array of same shape
    """

    # ✅ Convert NumPy → Torch
    if isinstance(eeg_signal, np.ndarray):
        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float32)

    eeg_signal = eeg_signal.to(device)

    with torch.no_grad():
        output = model(eeg_signal)

    # ✅ Convert Torch → NumPy
    return output.cpu().numpy()



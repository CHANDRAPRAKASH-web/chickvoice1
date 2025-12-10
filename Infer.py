# infer.py

import os
import torch
import numpy as np
import librosa

from model import ChickenResNet   # same architecture as training


# ---------- SETTINGS (must match training) ----------
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
CHECKPOINT_PATH = "models/chicken_resnet_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------


def load_model_and_labels(checkpoint_path: str, device: str):
    """
    Loads trained model checkpoint along with label mapping.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}\n"
            "Train the model to generate weights."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    label2idx = checkpoint["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    # build model
    num_classes = len(label2idx)
    model = ChickenResNet(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    return model, idx2label


def preprocess_wav(
    path: str,
    sample_rate: int = SAMPLE_RATE,
    duration: float = DURATION,
    n_mels: int = N_MELS
) -> torch.Tensor:
    """
    Convert WAV → fixed length → mel spectrogram → normalized tensor.
    """

    # load audio
    audio, _ = librosa.load(path, sr=sample_rate)

    # pad/crop to fixed duration
    target_len = int(sample_rate * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize (same as dataset.py)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # [1, 1, n_mels, time]
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor


def predict_file(audio_path: str):
    """
    Run inference on a single .wav file.
    """
    print(f"Loading model checkpoint: {CHECKPOINT_PATH}")
    model, idx2label = load_model_and_labels(CHECKPOINT_PATH, DEVICE)
    print("Class mapping:", idx2label)

    print(f"Processing file: {audio_path}")
    x = preprocess_wav(audio_path)
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    pred_conf = float(probs[pred_idx])

    print("\n======= PREDICTION =======")
    print(f"File        : {audio_path}")
    print(f"Predicted   : {pred_label}")
    print(f"Confidence  : {pred_conf:.3f}")
    print("All classes:")
    for i, p in enumerate(probs):
        print(f"  {idx2label[i]}: {p:.3f}")
    print("==========================\n")


if __name__ == "__main__":
    AUDIO_PATH = r"data\healthy\1.wav"   # <-- change this to test any file

    print("Running on device:", DEVICE)

    if not os.path.exists(AUDIO_PATH):
        print("ERROR: Set a valid WAV file in AUDIO_PATH.")
    else:
        predict_file(AUDIO_PATH)

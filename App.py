# app.py

import os
import io
import numpy as np
import librosa
import librosa.display
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ChickenResNet


# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
CHECKPOINT_PATH = "models/chicken_resnet_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------


@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f"Model not found at: {CHECKPOINT_PATH}")
        st.stop()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    label2idx = checkpoint["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    model = ChickenResNet(num_classes=len(label2idx))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, idx2label


def preprocess_audio(file_like):
    """Convert uploaded .wav into mel spectrogram tensor."""
    audio, sr = librosa.load(file_like, sr=SAMPLE_RATE)

    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    mel_tensor = torch.tensor(mel_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return mel_tensor, audio, mel_db


def predict(model, idx2label, mel_tensor):
    mel_tensor = mel_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="Chicken Health Detector", page_icon="🐔", layout="centered")
    st.title("🐔 Chicken Voice Health Classification")
    st.write("Upload a **chicken audio (.wav)** file. The model will classify it as:")
    st.write("- 🟢 **Healthy**")
    st.write("- 🔴 **Unhealthy**")
    st.write("- 🟡 **Noise Detected**")

    model, idx2label = load_model_and_labels()

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file is not None:

        # ---------- 1. AUDIO PREVIEW ----------
        st.subheader("🔊 Audio Preview")
        file_bytes = uploaded_file.read()
        st.audio(file_bytes, format="audio/wav")

        audio_buf = io.BytesIO(file_bytes)

        # ---------- 2. PREPROCESS + PREDICT ----------
        st.subheader("⚙ Processing Audio...")
        mel_tensor, audio, mel_db = preprocess_audio(audio_buf)
        pred_label, confidence, probs = predict(model, idx2label, mel_tensor)

        confidence_pct = round(confidence * 100)

        # ---------- OUTPUT MESSAGE ----------
        st.subheader("📌 Prediction Result")

        if pred_label.lower() == "healthy":
            st.success(f"🟢 Chicken is **Healthy** ({confidence_pct}%)")
        elif pred_label.lower() == "unhealthy":
            st.error(f"🔴 Chicken is **Unhealthy** ({confidence_pct}%)")
        else:
            st.warning(f"🟡 **Noise Detected** ({confidence_pct}%)")

        # ---------- 3. PROBABILITY BAR CHART ----------
        st.subheader("📊 Class Probabilities (%)")
        prob_dict = {
            idx2label[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }
        st.bar_chart(prob_dict)

        # ---------- 4. MEL SPECTROGRAM DISPLAY ----------
        st.subheader("📈 Mel Spectrogram")
        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(
            mel_db,
            sr=SAMPLE_RATE,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

        # ---------- RAW VALUES ----------
        st.subheader("📄 Raw Probability Data")
        st.json(prob_dict)

    else:
        st.info("👆 Upload a `.wav` file to start prediction.")


if __name__ == "__main__":
    main()

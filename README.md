# 🎧 Audio Deepfake Detection – ResMax Implementation

This repository provides a high-performance implementation of the **ResMax architecture** for detecting synthetic speech (audio deepfakes). Specifically optimized for the **ASVspoof 2019 Logical Access (LA)** dataset, the model utilizes **Residual Networks + Max Feature Map (MFM)** activation to deliver state-of-the-art detection with minimal computational overhead.

---

## 🚀 Project Overview

The ResMax approach effectively addresses the challenge of identifying subtle spectral artifacts introduced by modern **TTS (Text-to-Speech)** and **VC (Voice Conversion)** systems.  
By employing a competitive activation strategy, the model is:

- Robust against unseen attacks  
- Efficient enough for real-time deployment  

---

## 📊 Key Performance Metrics

| Metric | Value |
|--------|-------|
| Equal Error Rate (EER) | ~3.2% |
| Classification Accuracy | 97.3% |
| AUC (ROC) | 0.991 |
| Inference Latency | ~0.2s / 4s audio (GPU) |
| Model Size | ~15 MB |

---

## 🏗️ Technical Architecture

### 1️⃣ Max Feature Map (MFM) Activation

Unlike ReLU, MFM performs competitive selection:

```math
f(x_i, x_j) = \max(x_i, x_j)
Why MFM helps

Acts as winner-takes-all

Reduces feature maps by 50%

Highlights discriminative artifacts

2️⃣ Residual Blocks with MFM

Standard ResNet blocks

ReLU replaced with MFM layers

Ensures smooth gradient flow + compact representation

3️⃣ Feature Extraction Pipeline
Parameter	Value
Sampling Rate	16 kHz
Window / Hop	1024 / 256
Normalization	Log-power spectrogram
Input Shape	(250, 513, 1) → 4s audio
📁 Repository Structure
File	Description
main.py	Training & evaluation entry point
model.py	ResMax + MaxFeatureMap implementation
feature_extraction.py	STFT feature pipeline
data_generator.py	tf.data pipeline
inference.py	Single file prediction
evaluate.py	EER, AUC, Precision, Recall
data_utils.py	Dataset & protocol parsing
eda.py	Data analysis scripts
🛠️ Setup & Installation
✅ Prerequisites

Python 3.8+

CUDA GPU (recommended)

⚡ Quick Start
git clone https://github.com/your-repo/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install tensorflow==2.5.0 librosa==0.8.1 numpy pandas matplotlib scikit-learn

🔍 Dataset Preparation
LA/
└── LA/
    ├── ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019_LA_eval/flac/
    └── ASVspoof2019_LA_cm_protocols/

🏋️ Training
python main.py

🔮 Inference
from inference import predict_audio
import tensorflow as tf

model = tf.keras.models.load_model("resmax_model.h5")

result = predict_audio(model, "path/to/audio.flac")
print(f"Result: {result['prediction']} (Spoof Prob: {result['spoof_probability']:.4f})")

📈 Results & Observations

Increasing dataset size from 40% → 100% significantly reduced EER

Strong generalization to unseen attacks

Future scope:

Transformer-based temporal modeling

Multi-resolution spectrogram fusion

📚 Reference

Inspired by:
"ResMax: Detecting Voice Spoofing with Residual Network and Max Feature Map"


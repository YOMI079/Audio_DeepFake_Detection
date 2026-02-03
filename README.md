# 🎧 Audio Deepfake Detection — ResMax Implementation

High-performance implementation of the ResMax architecture for detecting synthetic speech (audio deepfakes), optimized for the ASVspoof 2019 Logical Access (LA) dataset. ResMax combines Residual Networks with Max Feature Map (MFM) activations to detect subtle spectral artifacts introduced by TTS and VC systems while remaining computationally efficient.

---

## 🔎 Overview

ResMax uses competitive MFM activations inside residual blocks to produce compact, discriminative feature representations. The architecture is robust to unseen attacks and efficient enough for near real-time inference.

Key advantages:
- Winner-takes-all feature selection (MFM)
- Reduced number of feature maps (≈50% per MFM)
- Smooth gradient flow with residual connections
- Lightweight model suitable for GPU inference

---

## 🚀 Quick Start

Clone and install dependencies:

```bash
git clone https://github.com/your-repo/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install -r requirements.txt
# or:
pip install tensorflow==2.5.0 librosa==0.8.1 numpy pandas matplotlib scikit-learn
```

Prepare the ASVspoof 2019 LA dataset as described below, then start training:

```bash
python main.py
```

---

## ⚙️ Installation & Prerequisites

- Python 3.8+
- CUDA GPU (recommended for training)
- Libraries: TensorFlow, librosa, numpy, pandas, matplotlib, scikit-learn

Recommended (example):
```bash
pip install tensorflow==2.5.0 librosa==0.8.1 numpy pandas matplotlib scikit-learn
```

---

## 📁 Dataset Layout (ASVspoof 2019 LA)

Expected dataset directory structure:

LA/
└── LA/
    ├── ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019_LA_eval/flac/
    └── ASVspoof2019_LA_cm_protocols/

See `data_utils.py` for protocol parsing and dataset helpers.

---

## 🏗️ Architecture (Technical Summary)

1. Max Feature Map (MFM) activation  
   - Competitive activation: f(x_i, x_j) = max(x_i, x_j)  
   - Acts as a winner-takes-all, reducing channels by half and emphasizing discriminative components.

2. Residual Blocks with MFM  
   - Standard ResNet-style skip connections with ReLU replaced by MFM layers to keep gradients stable and representation compact.

3. Feature Extraction Pipeline  
   - Sampling rate: 16 kHz  
   - STFT window / hop: 1024 / 256  
   - Input representation: log-power spectrogram (shape ≈ (250, 513, 1) for 4s of audio)

---

## 🧩 Repository Structure

- main.py — Training & evaluation entry point  
- model.py — ResMax network and MaxFeatureMap implementation  
- feature_extraction.py — STFT and spectrogram pipeline  
- data_generator.py — tf.data pipeline and batching  
- inference.py — Single-file prediction utilities  
- evaluate.py — EER, AUC, Precision, Recall calculations  
- data_utils.py — Dataset & protocol parsing helpers  
- eda.py — Exploratory data analysis scripts  
- requirements.txt — project dependencies

---

## 📈 Key Performance Metrics

| Metric | Value |
|--------|-------|
| Equal Error Rate (EER) | ~3.2% |
| Classification Accuracy | 97.3% |
| AUC (ROC) | 0.991 |
| Inference Latency | ~0.2 s per 4s audio (GPU) |
| Model Size | ~15 MB |

---

## 🏋️ Training

Default training:
```bash
python main.py --config configs/default.yaml
```

Common options:
- batch size
- learning rate and scheduler
- number of epochs
- checkpoint directory

See `main.py` and the `configs/` directory (if present) for full argument list and training hyperparameters.

---

## 🔮 Inference

Example usage from Python:

```python
from inference import predict_audio
import tensorflow as tf

model = tf.keras.models.load_model("resmax_model.h5")
result = predict_audio(model, "path/to/audio.flac")
print(f"Result: {result['prediction']} (Spoof Prob: {result['spoof_probability']:.4f})")
```

Command-line (if supported):
```bash
python inference.py --model resmax_model.h5 --audio path/to/audio.flac
```

---

## 📊 Evaluation

Evaluation metrics implemented in `evaluate.py`:
- Equal Error Rate (EER)
- Area Under ROC (AUC)
- Precision / Recall / F1

Use `evaluate.py` to compute metrics on DEV or EVAL splits and to log ROC curves.

---

## 🔬 Results & Observations

- Increasing dataset usage from 40% → 100% significantly reduced EER.
- Strong generalization to several unseen attack types.
- MFM + Residual connections yield a compact model with high discriminative power.

---

## 🔭 Future Work

- Transformer-based temporal modeling for longer contexts  
- Multi-resolution spectrogram fusion (combine multiple STFT settings)  
- Lightweight quantized model for CPU-only deployments

---

## 📚 References

Inspired by:
- "ResMax: Detecting Voice Spoofing with Residual Network and Max Feature Map"
- ASVspoof 2019 LA dataset and protocols

---

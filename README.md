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


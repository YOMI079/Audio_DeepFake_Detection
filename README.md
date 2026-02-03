Audio Deepfake Detection - ResMax Implementation

This repository provides a high-performance implementation of the ResMax architecture for detecting synthetic speech (audio deepfakes). Specifically optimized for the ASVspoof 2019 Logical Access (LA) dataset, the model utilizes Residual Networks coupled with Max Feature Map (MFM) activation to deliver state-of-the-art detection with minimal computational overhead.

🚀 Project Overview

The ResMax approach effectively addresses the challenge of identifying subtle spectral artifacts introduced by modern TTS (Text-to-Speech) and VC (Voice Conversion) systems. By employing a competitive activation strategy, the model is both robust against unseen attacks and efficient enough for real-time deployment.

Key Performance Metrics

Equal Error Rate (EER): ~3.2%

Classification Accuracy: 97.3%

Area Under ROC Curve (AUC): 0.991

Inference Latency: ~0.2s per 4-second clip (GPU)

Model Footprint: ~15 MB

🏗️ Technical Architecture

1. Max Feature Map (MFM) Activation

Unlike standard ReLU activation which simply thresholds values at zero, MFM promotes competitive feature selection. For a given input pair, only the maximum value is retained:


$$f(x_i, x_j) = \max(x_i, x_j)$$


This mechanism acts as a "winner-takes-all" filter, naturally reducing the number of feature maps by 50% while emphasizing the most discriminative artifacts.

2. Residual Blocks with MFM

The core of the network consists of residual blocks where the traditional non-linearity is replaced by MFM layers. This ensures smooth gradient flow while maintaining the competitive selection process throughout the network depth.

3. Feature Extraction Pipeline

Raw audio is transformed into the frequency domain to expose manipulation artifacts:

Sampling Rate: 16,000 Hz

Window/Hop: 1024 / 256

Normalization: Log-scale power spectrograms

Input Shape: (250, 513, 1) representing 4 seconds of audio.

📁 Repository Structure

File

Description

main.py

Primary entry point for training and full-scale evaluation.

model.py

Implementation of custom MaxFeatureMap layer and ResMax architecture.

feature_extraction.py

Class-based utility for consistent STFT-based feature engineering.

data_generator.py

Optimized tf.data pipeline supporting stratified sampling.

inference.py

High-level API for predicting single files with visualization.

evaluate.py

Metric computation logic (EER, AUC, Precision/Recall).

data_utils.py

Protocol parsing and dataset path management.

eda.py

Scripts for analyzing class balance and audio distributions.

🛠️ Setup & Installation

Prerequisites

Python 3.8+

CUDA-enabled GPU (Recommended for training)

Quick Start

Clone the repository:

git clone [https://github.com/your-repo/audio-deepfake-detection.git](https://github.com/your-repo/audio-deepfake-detection.git)
cd audio-deepfake-detection


Install dependencies:

pip install tensorflow==2.5.0 librosa==0.8.1 numpy pandas matplotlib scikit-learn


🔍 Detailed Usage

Dataset Preparation

Ensure the ASVspoof 2019 dataset is structured as follows:

LA/
└── LA/
    ├── ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019_LA_eval/flac/
    └── ASVspoof2019_LA_cm_protocols/


Training the Model

To train with the progressive fine-tuning strategy (defaulting to 40% sampling):

python main.py


Running Inference

To check a specific audio file for spoofing artifacts:

from inference import predict_audio
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('resmax_model.h5')

# Predict
result = predict_audio(model, "path/to/my_audio.flac")
print(f"Result: {result['prediction']} (Spoof Prob: {result['spoof_probability']:.4f})")


📈 Results Analysis

The model exhibits strong generalization capabilities. During development, scaling the dataset from 40% to 100% resulted in a significant drop in EER, highlighting the importance of diversity in deepfake training data. Future work includes implementing transformer-based sequential modeling and multi-resolution spectrogram inputs.

Implementation inspired by: "ResMax: Detecting Voice Spoofing with Residual Network and Max Feature Map"

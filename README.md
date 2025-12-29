AI-Generated Image Detection System
Dual-Stream Deep Learning with Explainability & GPU Acceleration
1. Introduction

With the rapid advancement of generative models such as GANs and diffusion models, AI-generated images have become increasingly realistic and widespread. This poses serious challenges in areas such as misinformation detection, digital forensics, and content authentication.

This project presents an AI Image Detector that classifies images as Real or Fake (AI-generated) using a dual-stream deep learning architecture combined with high-frequency feature extraction and explainability techniques. The system is trained and executed using GPU acceleration via WSL (Ubuntu) and PyTorch Nightly to support modern NVIDIA GPUs.

2. Objectives

The primary objectives of this project are:

Detect AI-generated images with high accuracy.

Capture both semantic features and low-level artifacts left by generative models.

Ensure GPU-accelerated training and inference.

Provide visual explanations using Grad-CAM.

Build a reproducible and shareable ML pipeline.

3. Dataset Description
Directory Structure
Project_Data/
│
├── train/
│   ├── 0_real/
│   └── 1_fake/
│
├── val/
│   ├── 0_real/
│   └── 1_fake/


0_real → Authentic camera images

1_fake → AI-generated images

The dataset was transferred from Windows to WSL for faster Linux-based GPU processing.

4. Preprocessing & Data Augmentation
Training Transformations

Resize to 224×224

Random Horizontal Flip

Random Rotation

Color Jitter

Normalization (ImageNet stats)

Validation / Inference Transformations

Resize to 224×224

Normalization only

This improves generalization while preventing overfitting.

5. Model Architecture
5.1 High-Pass Residual Extraction

A median filter–based residual is computed to isolate high-frequency noise patterns, which often reveal generative artifacts invisible to humans.

5.2 Dual-Stream Network

The model consists of two parallel CNN backbones:

Stream 1: Xception Network

Captures deep semantic features

Excellent at detecting subtle spatial inconsistencies

Stream 2: DenseNet-121

Encourages feature reuse

Preserves fine-grained texture information

5.3 Feature Fusion

Global Average Pooling on both streams

Feature concatenation

Fully connected classifier head

This fusion allows the model to analyze both global structure and local artifacts simultaneously.

6. Training Strategy

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Batch Size: GPU-optimized

Epochs: 10

Device: NVIDIA RTX 5050 Laptop GPU (WSL)

Observed Results

Epoch 1:

Training Loss ≈ 0.16

Validation Accuracy ≈ 97%

Training remained stable across epochs

7. Explainability with Grad-CAM

Grad-CAM was applied to the final convolutional layers to:

Highlight regions influencing predictions

Verify the model focuses on meaningful artifacts

Improve trust and interpretability

Heatmaps clearly showed attention on texture irregularities, unnatural edges, and synthetic patterns.

8. GPU Acceleration & Environment
Platform

Windows 11 + WSL2 (Ubuntu)

NVIDIA RTX 5050 Laptop GPU

Framework

PyTorch Nightly (CUDA-enabled)

Python 3.10

GPU utilization appeared moderate (10–20%) due to:

I/O overhead

Data loading

Lightweight forward passes

This is normal and expected for CNN workloads.

9. Deployment & Collaboration
Version Control

GitHub repository created

Large model file handled using Git LFS

SSH authentication configured

Collaboration

Collaborator added successfully

Repo includes:

Notebook

Model

Requirements

Execution guide

10. How to Run the Project
git clone https://github.com/Sachidanandabunu09/ai-image-detector.git
cd ai-image-detector
git lfs pull
pip install -r requirements.txt


Open ai_detector01.ipynb and run cells sequentially.

11. Key Challenges Solved

CUDA & GPU incompatibility on Windows

TensorFlow GPU limitations for new architectures

PyTorch compatibility with RTX 5050

GitHub 100MB file limit

SSH authentication setup

Git LFS integration

Each challenge was resolved using industry-standard practices.

12. Applications

AI-generated image detection

Digital forensics

Fake media verification

Content moderation

Research on generative artifacts

13. Future Improvements

Add Vision Transformer (ViT) stream

Multi-scale frequency analysis

Adversarial robustness testing

Web-based inference demo

Model quantization for deployment

14. Conclusion

This project demonstrates a real-world deep learning pipeline, from data handling to GPU-accelerated training, explainability, collaboration, and deployment. The dual-stream architecture effectively captures both semantic and artifact-based features, achieving strong performance and interpretability.

The system follows professional ML engineering standards and is suitable for academic, research, and industry use.

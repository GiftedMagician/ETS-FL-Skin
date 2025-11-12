# ETS-FL-Skin
### TEE–CPU Feature Partitioning and Incremental Transmission Driven Federated Learning Framework for Skin Disease Recognition

---

## 📘 Overview

**ETS-FL-Skin** is a federated learning framework designed for privacy-preserving and communication-efficient **skin disease recognition** in multi-institutional medical environments.  
It integrates **TEE–CPU feature partitioning** with an **incremental parameter transmission mechanism**, achieving secure computation, reduced communication cost, and high diagnostic accuracy.

This repository accompanies the paper:  
> *ETS-FL-Skin: A Federated Learning Framework for Skin Disease Recognition Driven by TEE–CPU Feature Partitioning and Incremental Transmission.*

---

## ⚙️ Key Features

- **TEE–CPU Feature Partitioning:**  
  Model is divided into trusted (TEE) and non-trusted (CPU) submodels for layered secure execution.
  
- **Incremental Parameter Transmission:**  
  Sparse and quantized gradients are transmitted between clients to reduce communication cost.

- **Adaptive Thresholding:**  
  Thresholds for sparsification are dynamically adjusted to balance accuracy and communication efficiency.

- **Privacy Risk Index (PRI):**  
  Quantifies privacy resistance based on PSNR between original and reconstructed images under gradient inversion attacks.

---

## 🧪 Experimental Setup

### Hardware & Environment
- CPU: **Intel Xeon Gold 5318Y**
- Trusted Execution: **Intel SGX 2.0**
- TEE Runtime: **Gramine**
- Frameworks: **PyTorch 2.1**, **gRPC**, **NumPy**, **Matplotlib**
- OS: **Ubuntu 22.04 LTS**

### Network Simulation
- Communication latency: **50–200 ms**
- Simulated with gRPC-based asynchronous messaging

### Dataset
- **ISIC-2019 Skin Lesion Dataset**  
  - 25,331 dermoscopic images  
  - 8 diagnostic categories (e.g., MEL, NV, BCC, AK, BKL, DF, VASC, SCC)  
  - Binary classification task: malignant vs. benign
  - Input size: 224 × 224 × 3  
  - Data split: 70% train / 15% validation / 15% test

---

## 🚀 Running the Experiments

### 1. Environment Setup
```bash
git clone https://github.com/yourusername/ETS-FL-Skin.git
cd ETS-FL-Skin
conda create -n etsfl python=3.8
conda activate etsfl
pip install -r requirements.txt

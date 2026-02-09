
---

## 📄 `requirements.txt`

```txt
# Core Python
python>=3.9

# Numerical & Data Handling
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2

# Visualization
matplotlib>=3.7
seaborn>=0.12

# Deep Learning - TensorFlow / Keras
tensorflow>=2.12
keras>=2.12

# Deep Learning - PyTorch
torch>=2.0
torchvision>=0.15
torchaudio>=2.0

# Image Processing
Pillow>=9.5
opencv-python>=4.8

# Utilities
tqdm>=4.65
psutil>=5.9

# Jupyter Environment
jupyterlab>=4.0
ipykernel>=6.25
```

✅ **Why this works**

* Covers **all notebooks**
* Compatible with **JupyterLab**
* No unnecessary packages (keeps grading environments stable)

---

## 📘 `README.md`

```md
# AI Capstone Project – Deep Learning for Land Classification

## Overview
This capstone project explores memory-efficient image loading, augmentation, and classification strategies using deep learning frameworks. The project focuses on satellite and geospatial imagery to classify **agricultural vs non-agricultural land** using CNNs and Vision Transformers (ViTs).

Both **Keras (TensorFlow)** and **PyTorch** implementations are developed and evaluated to compare performance, flexibility, and scalability.

---

## Project Objectives
- Implement memory-based vs generator-based image loading
- Build efficient data pipelines using Keras and PyTorch
- Train and evaluate CNN classifiers
- Compare Keras and PyTorch model performance
- Implement Vision Transformers (ViT)
- Develop hybrid CNN–Transformer architectures
- Evaluate models using accuracy, precision, recall, F1-score, and ROC curves

---

## Directory Structure
```

AI_Capstone_Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── images/
│       ├── agricultural/
│       └── non_agricultural/
│
├── notebooks/
│   ├── Module_1/
│   ├── Module_2/
│   └── Module_3/
│
├── models/
├── reports/
├── utils/
├── requirements.txt
└── README.md

````

---

## Modules Breakdown

### Module 1 – Data Engineering
- Memory-based vs generator-based data loading
- Image augmentation using Keras
- Custom dataset and dataloader implementation in PyTorch

### Module 2 – Model Training & Evaluation
- CNN training using Keras
- CNN training using PyTorch
- Comparative analysis of both frameworks

### Module 3 – Advanced Architectures
- Vision Transformers (ViT) in Keras
- Vision Transformers (ViT) in PyTorch
- CNN + Transformer hybrid model evaluation

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Setup Instructions

### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch JupyterLab

```bash
jupyter lab
```

---

## Notes for Evaluation

* All notebooks run end-to-end without manual intervention
* Relative paths are used for datasets
* Random seeds are fixed for reproducibility
* Outputs and visualizations are preserved for grading

---

## Future Scope

* Multi-class land-use classification
* Transfer learning with pre-trained satellite models
* Distributed training for large-scale datasets
* Deployment using TensorFlow Serving or TorchServe

---

## Author

**Developer:** Kajal Dadas
**Email:** [kajaldadas149@gmail.com](mailto:kajaldadas149@gmail.com)
**Contact:** 7972244559

```




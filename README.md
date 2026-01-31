# 🧠 Brain Tumor Detection using MRI Images and CNN

An AI-based medical imaging project that detects the presence of brain tumors from MRI scans using Convolutional Neural Networks (CNN).  
This system is designed as an **AI-assisted diagnostic tool** to support medical professionals, not replace them.

---

## 🚀 Project Overview

Brain tumor detection is a critical task in medical diagnosis. Manual analysis of MRI scans is time-consuming and prone to human error.  
This project leverages **Deep Learning (CNN)** to automatically classify MRI images into:

- **Tumor**
- **No Tumor**

The model learns spatial features from MRI images and provides reliable predictions along with performance evaluation metrics.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 📂 Dataset

- **MRI Brain Tumor Dataset**
- Two classes:
  - `yes` → Tumor present
  - `no` → No tumor
- Images are preprocessed and normalized before training.

> Dataset can be obtained from Kaggle or medical imaging repositories.

---

## ⚙️ Project Workflow

1. Dataset loading and exploration
2. Image preprocessing
   - Resizing
   - Normalization
   - Data augmentation
3. CNN model building
4. Model training and validation
5. Performance evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC Curve
6. Prediction on unseen MRI images
7. Model saving for deployment

---

## 🏗️ CNN Architecture

- Convolutional Layers (ReLU)
- Max Pooling Layers
- Flatten Layer
- Fully Connected Dense Layers
- Sigmoid Activation (Binary Classification)

---

## 📊 Results & Evaluation

- Training and validation accuracy & loss visualization
- Confusion Matrix for class-wise performance
- ROC-AUC curve for classification reliability
- Sample MRI predictions displayed with labels

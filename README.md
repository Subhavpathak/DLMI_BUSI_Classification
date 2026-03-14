# Breast Ultrasound Image Classification using Deep Learning

This project implements a deep learning based classification system for breast ultrasound images using the **BUSI (Breast Ultrasound Images) dataset**. The goal of this project is to classify ultrasound images into three categories:

- Normal
- Benign
- Malignant

The project also analyzes the **class imbalance problem** in the dataset and evaluates different techniques to improve classification performance.

---

# Dataset

The dataset used in this project is the **BUSI Breast Ultrasound Dataset**, available on Kaggle.

Dataset link:
https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset

The dataset contains ultrasound images divided into three classes:

| Class | Description |
|------|-------------|
| Normal | No tumor present |
| Benign | Non-cancerous tumor |
| Malignant | Cancerous tumor |

Example class distribution:

Benign      ~437  
Malignant   ~210  
Normal      ~133  

The dataset is **imbalanced**, meaning some classes contain more samples than others. This imbalance can affect the performance of machine learning models.

---

# Project Workflow

The complete pipeline followed in this project is:

Dataset Loading  
↓  
Class Distribution Analysis  
↓  
Train / Validation / Test Split  
↓  
Image Preprocessing  
↓  
Model Training  
↓  
Evaluation  
↓  
Comparison of Techniques  

---

# Model Architecture

The model used in this project is **ResNet18**, a convolutional neural network pretrained on ImageNet.

Transfer learning is used by modifying the final fully connected layer to classify images into three categories:

Normal  
Benign  
Malignant  

---

# Image Preprocessing

Before training, the following preprocessing steps are applied:

- Resize images to 224 × 224
- Convert images to PyTorch tensors

For augmentation experiments, additional transformations are applied:

- RandomHorizontalFlip
- RandomRotation
- RandomAffine

These transformations help increase dataset diversity and improve model generalization.

---

# Handling Class Imbalance

Four different experiments were performed:

### 1. Baseline Model
The baseline model is trained without applying any class imbalance handling technique. This provides a reference performance.

### 2. Oversampling
Oversampling increases the sampling frequency of minority classes during training using **WeightedRandomSampler**.

### 3. Data Augmentation
Data augmentation artificially increases the diversity of the training dataset by applying transformations such as flipping, rotation and affine transformations.

### 4. Focal Loss
Focal Loss focuses the learning process on difficult samples and reduces the effect of easy samples. It is commonly used in imbalanced classification tasks.

---

# Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Results

| Method | Accuracy | F1 Score |
|------|------|------|
| Baseline |  0.863248 | 0.853466 |
| Oversampling | 0.854701 |  0.842426 |
| Augmentation | **0.880342** | **0.876538** |
| Focal Loss | 0.794872 | 0.779904 |

Observation:

- Data augmentation achieved the best performance.
- Oversampling slightly reduced performance due to repeated samples.
- Focal loss provided competitive results but did not outperform augmentation.

---

# Project Structure

project/
│
├── dataset/
├── busi_classification.ipynb
├── report_dlmi.pdf
└── README.md

---

# How to Run the Project

1. Clone the repository

git clone <https://github.com/Subhavpathak/DLMI_BUSI_Classification/>

2. Install required libraries

pip install torch torchvision scikit-learn pandas numpy pillow

3. Download the BUSI dataset from Kaggle and place it inside the project directory.

4. Run the notebook or Python script to train and evaluate the models.

---

# Conclusion

In this project, a deep learning based breast ultrasound classifier was developed using the BUSI dataset.

Multiple class imbalance handling techniques were explored including oversampling, data augmentation and focal loss.

Experimental results show that **data augmentation improves classification performance and helps the model generalize better on unseen data**.

---

# Future Work

Possible improvements include:

- Using larger architectures such as EfficientNet
- Applying advanced augmentation techniques
- Using ensemble models
- Applying attention-based models

---

# Author

Subhav Kumar

# MRI Generalization: Cross-Domain Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20Imaging-green)
![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-MRI-purple)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-CNNs-red)
![Domain Shift](https://img.shields.io/badge/Domain%20Shift-Cross--Domain%20Evaluation-yellow)
![Data Augmentation](https://img.shields.io/badge/Data%20Augmentation-CNN-lightgrey)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikit-learn)

## Overview

This project investigates the **cross-domain generalization of deep learning models for brain tumor classification from MRI images**.

Rather than evaluating models only on images drawn from the same dataset used for training, this project evaluates models on a **separate external dataset** to investigate how well they perform under **dataset/domain shift**.

The study compares several pretrained convolutional neural network (CNN) architectures and evaluates whether data augmentation can improve external-domain performance.

The project is designed as a reproducible experimental study, using multiple random seeds and reporting **mean ± standard deviation** rather than relying on a single training run.

---

## Research Question

> **How well do pretrained CNN architectures generalize to an unseen brain MRI dataset, and can data augmentation improve their cross-domain robustness?**

The project focuses on the difference between strong **within-domain performance** and actual **external-domain generalization**.

---

## Experimental Pipeline

```text
                    Dataset A
                (Source Domain)
                       │
                       ▼
             Data preprocessing
                       │
                       ▼
              Train / Validation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      ResNet18     DenseNet121   EfficientNet-B3
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                Dataset B
              External Domain
                       │
                       ▼
            Cross-domain evaluation
                       │
                       ▼
             Accuracy / Precision
             Recall / Macro F1
             Confusion Matrices
                       │
                       ▼
             Data augmentation
               on EfficientNet-B3
                       │
                       ▼
             Cross-domain comparison
```
---

## Dataset
### Source Dataset — Dataset A
The source dataset contains contrast-enhanced brain MRI images belonging to three tumor classes:
- Glioma
- Meningioma
- Pituitary
The dataset is divided into training and validation subsets. The validation set is used for model selection during training.
### External Dataset — Dataset B
A separate brain MRI dataset is used exclusively for external testing.
Dataset B is not used during model training or validation. This allows the project to measure performance under a domain shift between the source and target datasets.

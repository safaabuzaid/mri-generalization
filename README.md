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

**Status: Ongoing research project**

---

## Research Question

The project currently investigates:

> - How do different pretrained CNN architectures compare in cross-domain brain tumor MRI classification?
> - How stable are their external-domain results across different random seeds?
> - Can data augmentation improve the cross-domain generalization of the selected architecture?
> - Which tumor classes are most difficult to classify across domains?
  
**Further experiments will investigate whether uncertainty estimation can provide additional insight into model reliability under domain shift.**

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

---

## Current Progress

### 1. Baseline architecture comparison — Completed
ResNet18, DenseNet121, and EfficientNet-B3 were evaluated using three random seeds.
EfficientNet-B3 demonstrated the strongest mean external-domain performance among the evaluated architectures and was selected for further experimentation.
### 2. Data augmentation — Completed
Data augmentation is being investigated using EfficientNet-B3.
The current augmentation pipeline includes:
Random horizontal flipping
Random rotation
Small affine transformations
Mild brightness and contrast variation
Image resizing and normalization
The augmented model is evaluated using the same three random seeds as the baseline experiment.
Preliminary results indicate an improvement in external-domain performance compared with the baseline EfficientNet-B3 model. The experiment is being rerun with complete automated result logging to ensure reproducibility.
### 3. Uncertainty estimation — In progress 
The next stage will investigate uncertainty estimation using Monte Carlo Dropout (MC-Dropout).
The goal is to examine whether model uncertainty can help identify predictions that are less reliable under domain shift.

----

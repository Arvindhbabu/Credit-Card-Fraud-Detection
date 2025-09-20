# Credit Card Fraud Detection

![Credit Card Fraud Detection](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning. It employs a Random Forest Classifier on an imbalanced dataset, with techniques like SMOTE for oversampling the minority class. The model analyzes anonymized transaction features to classify transactions as fraudulent or legitimate, providing a robust solution for fraud prevention in financial systems.

## Problem Statement

Credit card fraud poses a significant financial risk to banks, merchants, and consumers, with fraudulent transactions often representing a small fraction of overall activity, leading to highly imbalanced datasets. This project aims to develop a machine learning model using historical transaction data—including anonymized features from principal component analysis (PCA), transaction time, amount, and class labels—to accurately detect fraudulent activities. By addressing class imbalance through techniques like SMOTE and employing classifiers such as Random Forest, the model seeks to minimize false positives and negatives, enhancing fraud detection precision while maintaining operational efficiency in real-time payment systems.

## Dataset

- **Source**: The dataset (`creditcard.csv`) is a standard credit card fraud dataset, often sourced from Kaggle, containing anonymized transactions labeled as fraudulent (Class=1) or non-fraudulent (Class=0).
- **Size**: Typically ~284,807 entries, with severe class imbalance (~0.17% fraud).
- **Key Features**:
  - Numerical: Time, V1-V28 (PCA-transformed features), Amount.
  - Target: Class (binary: 0 for legitimate, 1 for fraud).
- **Preprocessing Steps**:
  - Handling missing values with SimpleImputer.
  - Feature scaling using StandardScaler.
  - Addressing imbalance with SMOTE.
  - Train-test split.

The dataset is loaded and processed in the Jupyter notebook.

## Technologies Used

- **Programming Language**: Python 3.12+
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Machine Learning: `scikit-learn` (for train-test split, StandardScaler, RandomForestClassifier, metrics like confusion matrix and classification report)
  - Imbalance Handling: `imblearn` (for SMOTE)
- **Environment**: Jupyter Notebook (Colab compatible)

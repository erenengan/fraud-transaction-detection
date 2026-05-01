# Fraud Detection Project

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-1.7+-brightgreen.svg)](https://xgboost.readthedocs.io/en/stable/python/index.html)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview
This project demonstrates a **fraud detection pipeline** using a real-world dataset of payment transactions. The goal is to identify **fraudulent transactions** accurately, addressing challenges such as **class imbalance** and **feature heterogeneity**.

---

## Dataset
The dataset (`bs140513_032310.csv`) contains transaction-level records with key features including transaction type, amount, sender and receiver information, and fraud label (`isFraud`).  

---

## Environment & Libraries
- Python 3.10+  
- Libraries:
  - `pandas`, `numpy` — Data manipulation  
  - `seaborn`, `matplotlib` — Data visualization  
  - `scikit-learn` — Machine learning, preprocessing, metrics  
  - `imblearn` — Handling class imbalance (SMOTE)  
  - `xgboost` — Gradient boosting classifier  

## Steps / Workflow

### 1. Data Exploration (EDA)
- Count plots of fraud vs non-fraud transactions  
- Boxplots of transaction amounts by category  
- Histograms of fraudulent vs non-fraudulent transaction amounts  
- Grouped summaries of fraud by **gender** and **category**  

### 2. Data Cleaning & Preprocessing
- Dropped `zipcodeOri` and `zipMerchant` due to single unique values  
- Encoded categorical features (`category`, `gender`, etc.)  
- Checked for missing values and converted object columns to numeric codes  

### 3. Handling Class Imbalance
- Applied **SMOTE** to oversample minority class  
- Stratified train-test splits to maintain class proportions  

### 4. Model Training
Trained and compared several models:  
- **Logistic Regression** — baseline & interpretable  
- **Random Forest Classifier** — tree-based model  
- **XGBoost Classifier** — gradient boosting model  
- **Voting Ensemble** — combination of the above models  

### 5. Model Evaluation
Metrics used:  
- **ROC AUC** — model separability  
- **PR AUC** — robust for imbalanced datasets  
- **F1 Score** — balance of precision and recall  
- **Precision / Recall** — for detecting fraud specifically  
- **Confusion Matrix** — visualize errors  


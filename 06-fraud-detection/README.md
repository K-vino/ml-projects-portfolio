# 💳 Credit Card Fraud Detection

**Level**: 🟡 Intermediate  
**Type**: Binary Classification - Imbalanced Dataset  
**Dataset**: Credit Card Fraud Dataset

## 📋 Project Overview

This project detects fraudulent credit card transactions using machine learning. It focuses on handling highly imbalanced datasets where fraud cases are rare (~0.17% of transactions). Perfect for learning advanced techniques like SMOTE, cost-sensitive learning, and specialized evaluation metrics.

## 🎯 Objectives

- Handle severely imbalanced datasets
- Learn advanced sampling techniques (SMOTE, ADASYN)
- Master evaluation metrics for imbalanced data
- Implement cost-sensitive learning
- Apply anomaly detection techniques
- Understand business impact of false positives/negatives

## 📊 Dataset Information

Credit card transactions dataset with fraud labels.

### Features
- **Time**: Seconds elapsed between transactions
- **V1-V28**: PCA-transformed features (anonymized)
- **Amount**: Transaction amount
- **Class**: Fraud label (0=Normal, 1=Fraud)

### Challenge
- **Extreme Imbalance**: Only 0.17% fraud cases
- **High Stakes**: False negatives cost money, false positives annoy customers

## 🔍 Key Techniques

- **SMOTE**: Synthetic minority oversampling
- **Cost-Sensitive Learning**: Weighted loss functions
- **Anomaly Detection**: Isolation Forest, One-Class SVM
- **Threshold Optimization**: Precision-Recall trade-offs
- **Business Metrics**: Cost-benefit analysis

## 📈 Expected Results

- **Precision**: ~85-90% (minimize false alarms)
- **Recall**: ~75-85% (catch most fraud)
- **F1-Score**: ~80-87%
- **ROC-AUC**: ~95-98%

---

**🎯 Perfect for**: Learning imbalanced data techniques, anomaly detection

**⏱️ Estimated Time**: 4-5 hours

**🎓 Difficulty**: Intermediate with advanced imbalanced data concepts

# Advanced_AI_for_Data_Science
Carlos Iván Fonseca Mondragón | A01771689

# Predicting medical costs with Gradient Descent vs Ensemble Methods

## Dataset

- **insurance.csv**: Medical insurance dataset containing patient demographics (age, sex, BMI, children, smoking status, region) and associated medical charges.

## Implementation Files

- **main.py**: Manual implementation of linear regression using gradient descent from scratch.  
- **main_ml.py**: Bagging ensemble model using scikit-learn with 50 decision trees.

## Key Results

| Model | Test R² | Variance Explained |
|-------|---------|------------------|
| Linear Regression | 0.7534 | 75% |
| Bagging Ensemble | 0.8642 | 86% |
| **Performance Improvement** | — | 14.7% increase in predictive accuracy |

## Usage
Run main.py for manual linear regression implementation or main_ml.py for the Bagging ensemble model. Both scripts include comprehensive evaluation metrics and visualizations.

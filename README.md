# Employee Attrition & Department Prediction Model

## Overview
This project builds a **multi-output deep learning model** to predict:
1. **Employee Attrition** (Binary Classification: Yes/No)
2. **Department Assignment** (Multi-Class Classification)

The model uses **TensorFlow/Keras** and **Scikit-Learn** for data preprocessing and training.

---

## Features
- **Multi-output model** for predicting both Attrition & Department.
- **Feature scaling & encoding** for better model performance.
- **Softmax activation** for department classification.
- **Sigmoid activation** for attrition classification.
- **Evaluation metrics** include Accuracy, Precision, Recall, and F1-score.

---

## Dataset
- **Source:** HR Employee Attrition Dataset
- **Columns Include:**
  - `Age`, `JobSatisfaction`, `DistanceFromHome`, `OverTime`, `WorkLifeBalance`
  - `Attrition` (Target Variable 1)
  - `Department` (Target Variable 2)

---

## Model Architecture
- **Input Layer:** 10 selected features
- **Shared Hidden Layers:** 2 Dense layers (64 & 32 neurons, ReLU activation)
- **Branch 1 (Department Prediction):** Dense (16 neurons, ReLU) → Dense (Softmax)
- **Branch 2 (Attrition Prediction):** Dense (16 neurons, ReLU) → Dense (Sigmoid)

---

## Performance Metrics
- **Department Prediction:** Weighted F1-score
- **Attrition Prediction:** Precision, Recall, F1-score
- **Loss Functions:**
  - `categorical_crossentropy` for Department
  - `binary_crossentropy` for Attrition

---

## How to Run
1. Install dependencies:  
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib

## Run the Jupyter Notebook
jupyter notebook attrition.ipynb

## Train the model and evaluate performance
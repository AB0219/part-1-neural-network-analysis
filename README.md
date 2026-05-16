# Neural Network Fundamentals and Training Behavior Analysis

## Project Overview

This project focuses on building and analyzing a Feed-Forward Neural Network for a supervised learning problem using a customer churn dataset.

The objective of this assignment is to understand how neural networks learn using:

- Forward propagation
- Loss calculation
- Backpropagation
- Weight and bias updates

The project also includes hyperparameter experimentation and performance evaluation.


# Dataset Information

Dataset Used:
- `customer_churn_nn.csv`

Target Variable:
- `churn`
    - 1 = Customer Churned
    - 0 = Customer Retained

Feature Types:
- Numerical Features
- Categorical Features

Identifier Column:
- `customer_id` (removed before training)


# Tasks Performed

## Task 1: Dataset Understanding

Performed:
- Dataset loading
- Shape analysis
- Data type analysis
- Missing value check
- Statistical summary
- Target variable distribution visualization


## Task 2: Data Preprocessing

Performed:
- Removal of identifier column
- Missing value handling
- Label encoding of categorical variables
- Feature scaling using StandardScaler
- Train-test splitting


## Task 3: Neural Network Model Building

Built a Feed-Forward Neural Network using TensorFlow/Keras with:
- Input layer
- Hidden layers
- ReLU activation
- Sigmoid output layer
- Adam optimizer
- Binary crossentropy loss function


## Task 4: Training and Evaluation

Performed:
- Model training
- Accuracy and loss visualization
- Model evaluation on test data
- Confusion matrix generation
- Classification report generation


## Task 5: Hyperparameter Experimentation

Conducted multiple experiments by changing:
- Number of neurons
- Learning rate
- Batch size
- Number of epochs
- Activation functions

A comparison table was created to analyze model performance.


## Task 6: Final Reflection

Discussed:
- Role of weights and biases
- Importance of activation functions
- Learning rate effects
- Underfitting and overfitting


# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras


# Project Structure

part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── customer_churn_nn.csv
│
└── results/
    ├── model_comparison_table.csv
    └── evaluation_outputs.png


# Installation

Install all required libraries using:

```bash
pip install -r requirements.txt

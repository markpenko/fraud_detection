# Fraud Detection

Fraud detection using a Random Forest classifier.

## Problem

Financial institutions process millions of credit card transactions each day. A small percentage of these transactions are fraudulent, resulting in significant financial losses.

The goal of this project is to build a machine learning model that detects potentially fraudulent transactions.

## Dataset

**Credit Card Fraud Detection Dataset 2023**

The dataset used in this project can be downloaded from Kaggle:  
[Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023?select=creditcard_2023.csv)

> Note: The dataset is too large to include in this repository. Please download it manually and place it in the `data/` folder before running the scripts.

## Project Structure

fraud_detection
│
├── data/  
│   └── .gitkeep        # placeholder for dataset directory
│
├── notebooks/  
│   └── random_forest.ipynb   # exploratory analysis and model development
│
├── src/  
│   └── random_forest.py      # implementation of the Random Forest model
│
├── requirements.txt          # project dependencies
├── README.md                 # project documentation
└── .gitignore                # ignored files and directories

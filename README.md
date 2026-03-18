# Stock Market Direction Prediction (NIFTY50)

This project explores machine learning techniques to predict stock price movements using NIFTY50 historical data.

The goal is to understand financial prediction pipelines while learning core machine learning concepts.

## Dataset

NIFTY50 historical dataset

Location:
data/NIFTY50_all.csv

Features include:

Prev Close
Open
High
Low
Volume
VWAP

## Models Implemented

Regression Models

Linear Regression
Ridge Regression
Lasso Regression

Classification Models

Logistic Regression
Decision Tree

## Feature Engineering

Daily Returns
5 Day Moving Average (MA5)

Target variable:

1 → Next day price goes UP  
0 → Next day price goes DOWN

## Evaluation Metrics

Regression

R2 Score
RMSE

Classification

Accuracy
Precision
Recall
ROC-AUC


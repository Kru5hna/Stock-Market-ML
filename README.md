# NIFTY50 Stock Market Direction Prediction

A machine learning project that explores regression and classification techniques to predict NIFTY50 stock price movements using historical OHLCV data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Observations](#key-observations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Roadmap](#roadmap)

---

## Project Overview

This project builds an end-to-end ML pipeline on NIFTY50 historical data. The pipeline covers:

1. Raw data ingestion from CSV
2. Feature engineering (daily returns, 5-day moving average)
3. Train/test split (80/20)
4. Regression models to predict closing price
5. Classification models to predict next-day direction (UP = 1 / DOWN = 0)
6. Evaluation using R², RMSE, Accuracy, Precision, Recall, F1, and ROC-AUC

---

## Dataset

**File:** `data/NIFTY50_all.csv`

| Feature | Description |
|---|---|
| `Prev Close` | Previous day's closing price |
| `Open` | Opening price |
| `High` | Intraday high |
| `Low` | Intraday low |
| `Volume` | Trading volume |
| `VWAP` | Volume Weighted Average Price |

**Total samples:** ~120,000 rows (train: ~96,000 / test: ~24,000)

---

## Feature Engineering

Two engineered features are added before modelling:

**Daily Return**
```
Daily_Return = (Close - Prev_Close) / Prev_Close
```

**5-Day Moving Average (MA5)**
```
MA5 = rolling mean of Close over last 5 trading days
```

**Target variable (classification)**
```
target = 1  →  next day Close > today Close  (UP)
target = 0  →  next day Close ≤ today Close  (DOWN)
```

---

## Models Implemented

### Regression
| Model | Purpose |
|---|---|
| Linear Regression | Baseline price prediction |
| Ridge Regression | L2 regularisation to handle multicollinearity |
| Lasso Regression | L1 regularisation with implicit feature selection |

### Classification
| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline for direction prediction |
| Decision Tree | Non-linear splits, visualisable rules |

---

## Results

### Regression Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | -0.0013 | 0.0273 |
| Ridge Regression | -0.0012 | 0.0273 |
| Lasso Regression | ~0.0000 | 0.0273 |

> **Note:** Negative R² scores indicate the models perform worse than simply predicting the mean. The very low RMSE (~0.027) reflects that the target is expressed as a normalised return rather than a raw price. Predicting near-zero daily returns is an inherently noisy task — the mean is a surprisingly hard baseline to beat. Lasso performed best (R² closest to 0) by shrinking uninformative coefficients to zero. See [Key Observations](#key-observations).

---

### Classification Results

#### Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (DOWN) | 0.51 | 0.58 | 0.54 | 11,829 |
| 1 (UP) | 0.53 | 0.45 | 0.49 | 12,174 |
| **Accuracy** | | | **0.52** | **24,003** |

**ROC-AUC: 0.517**

---

#### Decision Tree

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (DOWN) | 0.50 | 0.33 | 0.39 | 11,829 |
| 1 (UP) | 0.51 | 0.68 | 0.58 | 12,174 |
| **Accuracy** | | | **0.51** | **24,003** |

**ROC-AUC: 0.50**

---

### Model Comparison Summary

| Model | Type | Primary Metric | Score |
|---|---|---|---|
| Lasso Regression | Regression | R² | ~0.000 |
| Ridge Regression | Regression | R² | -0.0012 |
| Linear Regression | Regression | R² | -0.0013 |
| Logistic Regression | Classification | ROC-AUC | 0.517 |
| Decision Tree | Classification | ROC-AUC | 0.500 |

> Logistic Regression edges out the Decision Tree on AUC (0.517 vs 0.500) and shows more balanced precision/recall across both classes, making it the stronger classifier in this experiment.

---

## Key Observations

### Regression — Predicting Returns is Hard
Negative R² means the models could not beat the mean predictor on the test set. This is a well-known result in financial ML — daily returns are extremely noisy with very low signal-to-noise ratio. Lasso performed best (R² closest to 0) because its L1 penalty discarded uninformative features rather than fitting to noise, which is meaningful in itself.

### Classification — Marginal Signal Exists
Both classifiers achieved just above 50% accuracy, with Logistic Regression reaching ROC-AUC of 0.517. While this looks small, in financial direction prediction even a consistent edge above 50% has theoretical value — the key question is whether it survives transaction costs and holds in future out-of-sample periods.

### Decision Tree Bias Toward UP
The Decision Tree heavily favoured predicting UP (class 1), with recall of 0.68 for UP vs only 0.33 for DOWN. This likely reflects a structural upward drift in the dataset (bull market periods dominating) that the tree exploited without learning a genuinely generalised pattern.

### Class Balance
The test set is nearly balanced (11,829 DOWN vs 12,174 UP), so accuracy is a reliable metric here and the ~51–52% results are not an artefact of class imbalance.

---

## Project Structure

```
NIFTY50-ML/
│
├── data/
│   └── NIFTY50_all.csv          # Raw historical data
│
├── notebooks/
│   └── exploration.ipynb        # EDA and visualisations
│
├── src/
│   ├── features.py              # Feature engineering pipeline
│   ├── regression.py            # Linear, Ridge, Lasso models
│   └── classification.py        # Logistic Regression, Decision Tree
│
├── outputs/
│   ├── decision_tree.png        # Tree visualisation
│   ├── roc_logistics.png        # ROC curve — Logistic Regression
│   └── roc_decision_tree.png    # ROC curve — Decision Tree
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nifty50-ml.git
cd nifty50-ml

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Usage

```bash
# Run the full pipeline
python src/features.py          # Engineer features from raw data
python src/regression.py        # Train and evaluate regression models
python src/classification.py    # Train and evaluate classification models
```

---

## Roadmap

Planned improvements to increase predictive performance:

- **Lagged features** — use previous N days' returns, volume changes, and price momentum instead of same-day values
- **Technical indicators** — RSI, MACD, Bollinger Bands, On-Balance Volume (OBV)
- **Time-series aware validation** — replace random `train_test_split` with `TimeSeriesSplit` to prevent future data leaking into training
- **Ensemble methods** — Random Forest, XGBoost, LightGBM for stronger non-linear modelling
- **Threshold tuning** — adjust classification decision threshold to optimise for precision or recall depending on strategy
- **Walk-forward validation** — simulate live trading conditions with rolling re-training windows
- **Hyperparameter tuning** — `GridSearchCV` for Decision Tree depth, Ridge/Lasso alpha

---

> **Note on train/test splitting:** Standard `train_test_split(shuffle=True)` (scikit-learn default) is not appropriate for time-series data as it allows future dates to appear in the training set. The correct approach is:
> ```python
> from sklearn.model_selection import TimeSeriesSplit
> tscv = TimeSeriesSplit(n_splits=5)
> for train_idx, test_idx in tscv.split(X):
>     X_train, X_test = X[train_idx], X[test_idx]
> ```

---

*Built as a learning project exploring ML fundamentals with financial time-series data.*
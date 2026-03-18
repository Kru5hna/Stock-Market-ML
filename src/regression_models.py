import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


def train_regression_models(X_train, X_test, y_train, y_test):

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {"R2": r2, "RMSE": rmse}

    return results
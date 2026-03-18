from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, roc_auc_score


def train_logistic_model(X_train, X_test, y_train, y_test):

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))

    print("ROC AUC:", roc_auc_score(y_test, probs))

    return model


def train_decision_tree(X_train, X_test, y_train, y_test):

    model = DecisionTreeClassifier(max_depth=3)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    return model
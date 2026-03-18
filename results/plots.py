import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
from sklearn.tree import plot_tree


def plot_roc(y_test, y_prob, model_name):

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve - {model_name}")

    plt.legend()

    plt.savefig(f"results/plots/roc_{model_name}.png")

    plt.close()

def save_tree(model, feature_names, model_name):

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(12,8))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Down", "Up"],
        filled=True
    )

    plt.title(f"Decision Tree - {model_name}")

    plt.savefig(f"results/plots/tree_{model_name}.png")

    plt.close()
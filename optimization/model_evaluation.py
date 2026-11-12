"""Evaluation utilities for multiclass classification."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import learning_curve


def evaluate_classifier(estimator, X_test, y_test):
    """Print and return useful multiclass evaluation metrics."""
    predictions = estimator.predict(X_test)
    result = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro",
                             zero_division=0),
        "weighted_f1": f1_score(y_test, predictions, average="weighted",
                                zero_division=0),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_test)
        try:
            result["roc_auc_ovr_macro"] = roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="macro"
            )
        except ValueError:
            result["roc_auc_ovr_macro"] = np.nan

    for key, value in result.items():
        print(f"\n{key}:\n{value}")
    return result


def plot_confusion_matrix(estimator, X_test, y_test):
    """Plot a confusion matrix using the estimator's class order."""
    predictions = estimator.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, xticks_rotation=45, cmap="Blues"
    )
    plt.tight_layout()
    plt.show()


def plot_learning_curve(estimator, X, y, cv=5, scoring="f1_macro"):
    """Plot training and cross-validation scores against training size."""
    sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.3, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    plt.plot(sizes, train_mean, marker="o", label="Training score")
    plt.fill_between(sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
    plt.plot(sizes, validation_mean, marker="o", label="Cross-validation score")
    plt.fill_between(
        sizes, validation_mean-validation_std,
        validation_mean+validation_std, alpha=0.15
    )
    plt.xlabel("Number of training samples")
    plt.ylabel(scoring)
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(estimator, X_test, y_test):
    """Plot one-vs-rest ROC curves when class probabilities are available."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("The estimator must support predict_proba().")
    from sklearn.preprocessing import label_binarize

    classes = estimator.classes_
    y_binary = label_binarize(y_test, classes=classes)
    probabilities = estimator.predict_proba(X_test)

    for index, class_name in enumerate(classes):
        if len(np.unique(y_binary[:, index])) < 2:
            continue
        RocCurveDisplay.from_predictions(
            y_binary[:, index], probabilities[:, index],
            name=f"Class {class_name} vs rest"
        )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title("Multiclass ROC (one-vs-rest)")
    plt.legend()
    plt.tight_layout()
    plt.show()

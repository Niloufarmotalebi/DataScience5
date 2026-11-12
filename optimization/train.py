"""End-to-end experiment: split, compare ensembles, tune a pipeline."""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from optimization.data_processor import DataProcessor
from models.ensemble import vote_hard, vote_soft, make_bagging_or_boosting
from evaluation.model_evaluation import (
    evaluate_classifier, plot_confusion_matrix, plot_learning_curve
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/merged_dataset.csv")
    parser.add_argument("--target", default="Histology")
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=0)
    if args.target not in data.columns:
        raise KeyError(f"Target {args.target!r} not found in {data.columns.tolist()}")

    y = data[args.target].astype(str)
    X = data.drop(columns=[args.target])
    # Identifier columns are not predictive features.
    X = X.drop(columns=[
        col for col in X.columns
        if col.lower() in {"patient_id", "sample_id", "case_id", "geo_accession"}
    ], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipeline: preprocessing and feature selection are fitted inside each fold.
    pipeline = Pipeline([
        ("preprocess", DataProcessor()),
        ("select", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(
            max_iter=3000, class_weight="balanced"
        ))
    ])
    parameter_grid = {
        "select__k": [5, 10, 20, 40],
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs"],
    }
    search = GridSearchCV(
        pipeline, parameter_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Best CV macro-F1:", search.best_score_)
    print("\nTuned pipeline test results:")
    evaluate_classifier(search.best_estimator_, X_test, y_test)
    plot_confusion_matrix(search.best_estimator_, X_test, y_test)
    plot_learning_curve(search.best_estimator_, X_train, y_train, cv=cv)

    # Compare hard voting, soft voting and boosting on the same set.
    for name, model in [
        ("Hard voting", vote_hard()),
        ("Soft voting", vote_soft()),
        ("AdaBoost", make_bagging_or_boosting()),
    ]:
        print(f"\n===== {name} =====")
        model.fit(X_train, y_train)
        evaluate_classifier(model, X_test, y_test)


if __name__ == "__main__":
    main()

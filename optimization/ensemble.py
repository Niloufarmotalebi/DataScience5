"""Hard/soft voting and baseline ensemble comparisons."""
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from optimization.data_processor import DataProcessor


def make_base_estimators(random_state=42):
    """Return three diverse classifiers, each with its own preprocessing."""
    return [
        ("logistic", Pipeline([
            ("preprocess", DataProcessor()),
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced"
            ))
        ])),
        ("random_forest", Pipeline([
            ("preprocess", DataProcessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=random_state
            ))
        ])),
        ("svc", Pipeline([
            ("preprocess", DataProcessor()),
            ("classifier", SVC(
                probability=True, class_weight="balanced",
                random_state=random_state
            ))
        ])),
    ]


def vote_hard(estimators=None):
    """Majority-vote classifier."""
    estimators = estimators or make_base_estimators()
    return VotingClassifier(estimators=estimators, voting="hard")


def vote_soft(estimators=None):
    """Average predicted class probabilities and choose the largest."""
    estimators = estimators or make_base_estimators()
    return VotingClassifier(estimators=estimators, voting="soft")


def make_bagging_or_boosting(random_state=42):
    """Return an AdaBoost comparison model with preprocessing."""
    return Pipeline([
        ("preprocess", DataProcessor()),
        ("classifier", AdaBoostClassifier(n_estimators=50,
                                          random_state=random_state))
    ])

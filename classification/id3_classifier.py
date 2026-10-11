import numpy as np
import pandas as pd


def entropy(y):
    """Calculate the entropy of the target distribution."""
    targets = np.unique(y)

    distribution = []

    for target in targets:
        count = 0

        for sample in y:
            if sample == target:
                count += 1

        distribution.append(count)

    probs = []

    for i in distribution:
        frequency = i / len(y)
        probs.append(frequency)

    result = 0

    for prob in probs:
        result += -(prob * np.log2(prob))

    return result


def information_gain(data, feature, target):
    """Calculate information gain for a feature."""
    entire_entropy = entropy(data[target])
    unique_values = data[feature].unique()
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]

        subset_entropy = entropy(subset[target])

        weight = len(subset) / len(data)

        weighted_entropy += weight * subset_entropy

    ig = entire_entropy - weighted_entropy

    return ig


def id3(data, features, target, depth=0, max_depth=None):
    """Create a decision tree using the ID3 algorithm."""

    if data[target].nunique() == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode()[0]

    if max_depth is not None and depth >= max_depth:
        return data[target].mode()[0]

    gains = {}

    for feature in features:
        gain = information_gain(data, feature, target)
        gains[feature] = gain

        if depth == 0:
            print(f"feature {feature} gain {gain:.2f}")

    best_feature = max(gains, key=gains.get)
    best_gain = gains[best_feature]

    if depth == 0:
        print(f"best feature: {best_feature}")

    if best_gain <= 0:
        return data[target].mode()[0]

    tree = {best_feature: {}}

    remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]

        tree[best_feature][value] = id3(
            subset,
            remaining_features,
            target,
            depth + 1,
            max_depth
        )

    return tree


def predict_one(tree, sample, default):
    """Predict the class label for one sample."""
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample[feature]

        if value in tree[feature]:
            tree = tree[feature][value]
        else:
            return default

    return tree


def predict_id3(tree, X_test, default):
    """Predict class labels for multiple samples."""
    return [
        predict_one(tree, sample, default)
        for _, sample in X_test.iterrows()
    ]


class ID3Classifier:
    """Decision tree classifier using the ID3 algorithm."""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            training_data = X.reset_index(
                drop=True
            ).copy()
        else:
            X = np.asarray(X)

            self.feature_names_ = [
                f"feature_{i}" for i in range(X.shape[1])
            ]

            training_data = pd.DataFrame(
                X,
                columns=self.feature_names_
            )

        self.target_name_ = "__target__"
        training_data[self.target_name_] = np.asarray(y)

        self.default_class_ = pd.Series(y).mode()[0]

        self.tree_ = id3(
            training_data,
            self.feature_names_,
            self.target_name_,
            max_depth=self.max_depth
        )

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.feature_names_
            )

        return np.asarray(
            predict_id3(
                self.tree_,
                X,
                self.default_class_
            )
        )

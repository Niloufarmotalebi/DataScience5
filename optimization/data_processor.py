"""Scikit-learn compatible preprocessing for mixed clinical/genomic data."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


class DataProcessor(BaseEstimator, TransformerMixin):
    """Impute, encode and scale clinical/genomic predictors.

    Numeric columns are median-imputed and standardized. Categorical columns
    are most-frequent-imputed and one-hot encoded. A low-variance filter is
    applied after preprocessing. The transformer learns all statistics only
    from the training fold when used inside a Pipeline.

    Parameters
    ----------
    scale_numeric : bool, default=True
        Standardize numeric features.
    remove_constant : bool, default=True
        Remove constant transformed features.
    """

    def __init__(self, scale_numeric=True, remove_constant=True):
        self.scale_numeric = scale_numeric
        self.remove_constant = remove_constant

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_columns_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns_ = [
            col for col in X.columns if col not in self.numeric_columns_
        ]

        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers = []
        if self.numeric_columns_:
            transformers.append((
                "numeric", Pipeline(numeric_steps), self.numeric_columns_
            ))
        if self.categorical_columns_:
            transformers.append((
                "categorical", categorical_pipeline, self.categorical_columns_
            ))

        if not transformers:
            raise ValueError("No usable predictor columns were found.")

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )
        transformed = self.preprocessor_.fit_transform(X, y)

        self.variance_filter_ = None
        if self.remove_constant:
            self.variance_filter_ = VarianceThreshold(threshold=0.0)
            self.variance_filter_.fit(transformed)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.loc[:, self.feature_names_in_]
        transformed = self.preprocessor_.transform(X)
        if self.variance_filter_ is not None:
            transformed = self.variance_filter_.transform(transformed)
        return transformed

    def get_feature_names_out(self, input_features=None):
        names = self.preprocessor_.get_feature_names_out()
        if self.variance_filter_ is not None:
            names = names[self.variance_filter_.get_support()]
        return names

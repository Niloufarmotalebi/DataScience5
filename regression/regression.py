import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_cost(X, y, theta, lambda_):
    m = len(y)

    predictions = X.dot(theta)
    errors = predictions - y

    cost = np.sum(errors ** 2) / (2 * m)

    regularization = lambda_ * np.sum(theta[1:] ** 2) / (2 * m)

    J = cost + regularization
    return J


def gradient_descent(X, y, theta, alpha, num_iters, lambda_):

    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y

        gradient = X.T.dot(errors) / m

        gradient[1:] += (lambda_ / m) * theta[1:]

        theta = theta - alpha * gradient

        J_history[i] = compute_cost(X, y, theta, lambda_)
   
    return theta, J_history


def reverse_theta(theta, scaler):
    """
    Transforms the theta parameters obtained from a regression model on
    scaled data back to the original feature space.
    """
    theta_original = np.zeros_like(theta)
    #intercept term
    theta_original[0] = theta[0] - np.sum(
        (theta[1:] * scaler.mean_) / scaler.scale_
    )
    #coefficient terms
    theta_original[1:] = theta[1:] / scaler.scale_

    return theta_original


class LinearRegression:
    """
    Linear regression model using gradient descent.

    Parameters
    ----------
    alpha : float
        Learning rate.
    num_iters : int
        Number of gradient descent iterations.
    lambda_ : float
        Regularization parameter.
    """

    def __init__(self, alpha=0.01, num_iters=10000, lambda_=0.0):
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = None
        self.scaler = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the data.
        """

        # Scale the features
        self.scaler = StandardScaler()
        X_scaled_features = self.scaler.fit_transform(X)

        # Add intercept
        X_scaled = np.c_[
            np.ones((X_scaled_features.shape[0], 1)),
            X_scaled_features
        ]

        # Initialize theta
        theta = np.zeros(X_scaled.shape[1])

        # Gradient descent
        self.theta, self.cost_history = gradient_descent(
            X_scaled,
            y,
            theta,
            self.alpha,
            self.num_iters,
            self.lambda_
        )

        return self

    def predict(self, X):
        """
        Predict target values for new data.
        """

        X_scaled_features = self.scaler.transform(X)

        X_scaled = np.c_[
            np.ones((X_scaled_features.shape[0], 1)),
            X_scaled_features
        ]

        return X_scaled.dot(self.theta)

    def get_original_theta(self):
        """
        Return theta values in the original feature scale.
        """

        return reverse_theta(self.theta, self.scaler)

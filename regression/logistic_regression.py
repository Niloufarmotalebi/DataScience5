import numpy as np
from sklearn.preprocessing import StandardScaler


def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def compute_cost(X, y, theta, lambda_):
    """
    Compute cost for logistic regression with regularization.

    Parameters:
    X:  Input feature matrix (m x n)
    y: True labels vector (m,)
    theta: Parameters vector (n,)
    lambda_: Regularization parameter

    Return:
    J:Cost value
    """
    m = len(y)

    h = sigmoid(X.dot(theta))

    cost = -1 / m * (
        y.dot(np.log(h)) +
        (1 - y).dot(np.log(1 - h))
    )

    regularization = lambda_ / (2 * m) * np.sum(theta[1:] ** 2)

    J = cost + regularization

    return J


def gradient_descent(X, y, theta, alpha, num_iters, lambda_):
    """
    Perform gradient descent to find optimal theta.

    Parameters:
    X: Input feature matrix (m x n)
    y: True labels vector (m,)
    theta: Initial parameters vector (n,)
    alpha: Learning rate
    num_iters: Number of iterations
    lambda_:  Regularization parameter

    Return:
    theta:  Updated parameters vector
    J_history:  History of cost values
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):

        # Predictions
        h = sigmoid(X.dot(theta))

        # Error
        errors = h - y

        # Gradient
        gradient = X.T.dot(errors) / m

        # Regularization (do not regularize theta[0])
        gradient[1:] += (lambda_ / m) * theta[1:]

        # Update parameters
        theta = theta - alpha * gradient

        # Store cost
        J_history[i] = compute_cost(X, y, theta, lambda_)

    return theta, J_history


class LogisticRegression:
    """
    Logistic regression model using gradient descent.

    Parameters
    ----------
    alpha : float
        Learning rate.
    num_iters : int
        Number of gradient descent iterations.
    lambda_ : float
        Regularization parameter.
    """

    def __init__(
        self,
        alpha=0.01,
        num_iters=15000,
        lambda_=0.7,
        threshold=0.5
    ):
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.threshold = threshold
        self.theta = None
        self.cost_history = None
        self.scaler = None

    def fit(self, X, y):
        """
        Fit the logistic regression model to the data.
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

    def predict_proba(self, X):
        """
        Predict probability of the positive class.
        """

        X_scaled_features = self.scaler.transform(X)

        X_scaled = np.c_[
            np.ones((X_scaled_features.shape[0], 1)),
            X_scaled_features
        ]

        return sigmoid(X_scaled.dot(self.theta))

    def predict(self, X):
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters theta.
        """

        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)

        return predictions

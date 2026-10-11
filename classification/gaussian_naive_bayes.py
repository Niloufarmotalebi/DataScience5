import numpy as np


def get_labels(y_train):
    """Extract unique class labels from the training data."""
    labels = np.unique(y_train).tolist()
    return labels


def get_priors(y_train, labels):
    """Calculate prior probabilities for each unique label."""
    priors = []

    for label in labels:
        occurrence = np.sum(y_train == label)
        prior = occurrence / len(y_train)
        priors.append(prior)

    return priors


def get_likelihoods(X_train, y_train):
    """Calculate mean and standard deviation matrices."""
    labels = get_labels(y_train)
    n = X_train.shape[1]

    mu = np.zeros((len(labels), n))
    sd = np.zeros((len(labels), n))

    for i in labels:
        idx = np.where(y_train == i)[0]

        for j in range(n):
            mu[i][j] = np.mean(X_train[idx, j])
            sd[i][j] = np.std(X_train[idx, j])

    return mu, sd


def classify(X_test, labels, priors, mu, sd):
    """Classify each test sample."""
    y_pred = []

    for i in range(len(X_test)):
        posteriors = []

        for j in labels:
            squared_differences = (X_test[i, :] - mu[j, :])**2

            norm_factors = 1 / np.sqrt(
                2 * np.pi * sd[j, :]**2
            )

            likelihoods = norm_factors * np.exp(
                -squared_differences / (2 * sd[j, :]**2)
            )

            posterior = priors[j] * np.prod(likelihoods)
            posteriors.append(posterior)

        predicted_label = labels[np.argmax(posteriors)]
        y_pred.append(predicted_label)

    return y_pred


def evaluateNB(y_test, y_pred):
    """Calculate classification accuracy."""
    match = 0

    for i in np.arange(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            match = match + 1

    acc = match / len(y_test)
    return acc


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier."""

    def fit(self, X_train, y_train):
        self.labels_ = get_labels(y_train)
        self.priors_ = get_priors(y_train, self.labels_)
        self.mu_, self.sd_ = get_likelihoods(
            X_train, y_train
        )

        return self

    def predict(self, X_test):
        return np.asarray(
            classify(
                X_test,
                self.labels_,
                self.priors_,
                self.mu_,
                self.sd_
            )
        )

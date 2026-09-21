# Evaluation of Regression Models

## Introduction

Machine learning is a field in which algorithms learn patterns from data. Tom Mitchell (1997) defines machine learning as:

> “A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.”

For example, predicting a company's revenue is a **regression task**, while detecting spam emails is a **classification task**.

## Supervised Learning

Linear Regression and Logistic Regression are both **supervised learning** methods because the training data contains target values or labels.


## Linear Regression

Linear Regression is used to predict a **continuous numerical target**. It is suitable when the relationship between the features and the target can be reasonably approximated by a linear relationship.

In this assignment, Linear Regression is used to predict **logS**, the logarithm of aqueous solubility, from molecular features.

Another example is predicting blood glucose concentration from physiological measurements.

Linear Regression is generally not preferred for classification. Although it can technically be used with binary values such as 0 and 1, its predictions are not restricted to probabilities between 0 and 1.

It is also less suitable when the relationship between the features and target is strongly nonlinear.

## Logistic Regression

Logistic Regression is mainly used for **classification**, especially when there are two possible classes. It estimates the probability of belonging to a class and uses a threshold to assign a class.

For example, Logistic Regression can be used to predict whether a patient has a disease. A high predicted probability can be classified as positive (1), while a low probability can be classified as negative (0).

In this assignment, Logistic Regression is used to classify molecules as **active or inactive**.

The dataset comes from the **ChEMBL** database and contains molecular structures represented as **SMILES** and bioactivity values based on **IC50** measurements. The bioactivity is converted into two classes, active (1) and inactive (0). Molecular features are calculated from the SMILES structures using **RDKit**.

Logistic Regression is not designed to directly predict continuous numerical values. It is also less suitable when the relationship between the features and the outcome is strongly nonlinear.

## Comparison of the Models

| Model               | Learning type           | Target                     | Example                              |
| ------------------- | ----------------------- | -------------------------- | ------------------------------------ |
| Linear Regression   | Supervised, model-based | Continuous numerical value | Predicting molecular logS            |
| Logistic Regression | Supervised, model-based | Binary class               | Predicting active/inactive molecules |

Both models use a linear combination of input features, but they are designed for different prediction tasks.

## Evaluation

For Linear Regression, model performance is evaluated using **Mean Squared Error (MSE)**

For Logistic Regression, performance is evaluated using **accuracy**.

Both models use gradient descent and regularization. Regularization helps control the size of the model coefficients and can reduce overfitting.

## Conclusion

Linear Regression is suitable for continuous numerical targets, while Logistic Regression is mainly used for classification.

In this assignment, Linear Regression is used to predict molecular **logS**, while Logistic Regression is used for binary **active/inactive** classification.

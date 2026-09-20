# Evaluation of Regression Models

## Introduction

Different regression models are suitable for different types of data and research questions. The choice of model depends mainly on the type of target variable and on the relationship between the input features and the target. In this assignment, linear regression and logistic regression are considered.

The aim of this evaluation is to describe when each model is appropriate or inappropriate and to give examples of their use in data science and life sciences.

## Linear Regression

Linear regression is appropriate when the target variable is continuous and the relationship between the features and the target can be reasonably approximated by a linear relationship. The model predicts a continuous numerical value.

For example, linear regression can be used to predict the **logS (logarithm of aqueous solubility)** of a molecule from molecular features. In this situation, the target variable is continuous, making linear regression a suitable model to investigate whether the molecular features can explain differences in solubility.

Another example in life sciences is predicting a person's blood glucose level from variables such as carbohydrate intake, physical activity, or other physiological measurements. Since blood glucose is a continuous variable, linear regression can be considered if the relationships between the predictors and glucose level are approximately linear.

Linear regression is less appropriate when the target variable is categorical. For example, predicting whether a patient has a disease or does not have a disease is not a suitable problem for ordinary linear regression because the target consists of categories rather than a continuous value. It is also less suitable when the relationship between the features and the target is strongly nonlinear and cannot be reasonably approximated by a straight line.

## Logistic Regression

Logistic regression is appropriate when the target variable is categorical, especially when there are two possible classes. Instead of predicting a continuous value directly, logistic regression estimates the probability that an observation belongs to a particular class.

For example, logistic regression can be used to predict whether a molecule is **active or inactive** based on its molecular properties. The target has two possible classes, so logistic regression is suitable for this type of classification problem.

A similar example in life sciences is predicting whether a patient has a particular medical condition based on measured characteristics or biomarkers. The output can represent the probability of belonging to the positive class, which can then be converted into a class prediction.

Logistic regression is not appropriate when the target variable is continuous, such as predicting blood glucose concentration or molecular solubility. In addition, standard binary logistic regression is not sufficient by itself when there are more than two classes unless an appropriate multiclass extension is used.

Logistic regression can also be unsuitable when the relationship between the features and the class probability is highly nonlinear. In such cases, a more flexible classification model may represent the data better.

## Comparison of the Models

| Model | Target variable | Suitable relationship | Example in Life Sciences |
|---|---|---|---|
| Linear Regression | Continuous | Approximately linear | Predicting molecular logS |
| Logistic Regression | Categorical/binary | Relationship between features and log-odds is approximately linear | Predicting active/inactive molecules |

The main difference between these models is the type of prediction they are designed to make. Linear regression is used to predict continuous numerical values, while logistic regression is used for classification problems.

## Application to the Assignment

The **logS dataset** used in the assignment provides an example where linear regression is appropriate because logS is a continuous target variable. The performance of the regression model can be evaluated using measures such as **mean squared error (MSE)** and **R²**.

For a problem where the target represents two classes, such as whether a molecule is active or inactive, logistic regression would be more appropriate. In this situation, classification measures such as **accuracy, precision, recall, and a confusion matrix** can be used to evaluate the model.

## Conclusion

There is no single regression model that is appropriate for every dataset. Linear regression is useful for continuous targets with approximately linear relationships, while logistic regression is designed for categorical or binary outcomes.

Therefore, selecting a regression model should be based on both the type of target variable and the relationship between the predictors and the target. In a life-sciences context, understanding these differences is important because biological and molecular data can contain different types of relationships and prediction tasks.
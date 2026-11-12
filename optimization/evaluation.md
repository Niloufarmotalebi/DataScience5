# Project Files

### 1. load_data.py
This file is responsible for loading and combining the datasets. It reads the Clinical dataset, which contains patient information, and the Gene Expression dataset, which contains gene expression values. It matches the two datasets using patient/sample IDs and combines them into a single dataset. The target variable for the classification task is `Histology`.

### 2. data_processor.py
This file prepares the data before training the machine learning models. It handles missing numerical values using the median and missing categorical values using the most frequent value. It also scales numerical features using `StandardScaler`, converts categorical features into numerical values using `OneHotEncoder`, and removes constant features that have no variation.

### 3. model_evaluation.py
This file evaluates the performance of the trained machine learning models. It calculates different evaluation metrics, including Accuracy, Precision, Recall, F1-score, Balanced Accuracy, and ROC-AUC. It also generates a Confusion Matrix to show correct and incorrect predictions, Learning Curves to identify overfitting and underfitting, and ROC curves for multiclass classification.

### 4. train.py
This file handles the training and optimization of the machine learning models. It splits the dataset into training and testing sets, creates a Pipeline combining preprocessing, feature selection, and classification, and uses `SelectKBest` to select the most relevant features. It also uses `GridSearchCV` to find the best hyperparameters, trains Logistic Regression with the best parameters, evaluates the model on the test set, and compares Hard Voting, Soft Voting, and AdaBoost.

The code for this assignment was generated using AI. I have reviewed it to make sure it works correctly and accurately.
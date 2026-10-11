# Classification Models: Evaluation

## 1. Gaussian Naive Bayes

Gaussian Naive Bayes was implemented using custom functions and a `GaussianNaiveBayes` class with `fit()` and `predict()` methods. Its performance was compared with Scikit-learn's `GaussianNB` implementation using the Iris dataset.

### When it is appropriate
 When it is appropriate:
- When the dataset contains continuous numerical features.
- When All features are equally important and statistically independent

In Naive Bayes the likelyhood of each feature given the class follows a normal distribution, by class
specific mean μ and variance σ

### When it is not appropriate
- When features are strongly dependent
- When a feature has zero variance within a class, unless this case is handled appropriately.
- When complex relationships between features and class labels are needed to achieve good predictive performance.

### Evaluation
Accuracy was used to evaluate both models on the same test set. The custom model gives similar results to the Scikit-learn model on this dataset.
---
## 2. ID3 Decision Tree

ID3 is a decision tree algorithm that selects features based on information gain.
The custom classifier was implemented using `entropy()`, `information_gain()`, and `id3()` functions, and an `ID3Classifier` class with `fit()` and `predict()` methods.

### When it is appropriate
- When features are categorical or discrete.
- When features affect the target in complex ways.
- When a visual representation of the model is useful.

### When it is not appropriate
- When the tree becomes too complex.
- When categorical features have many unique values, which can make information gain favor certain features.

### Evaluation
The custom ID3 classifier was evaluated using the METABRIC clinical dataset to predict patient vital status, represented by the classes `alive` and `died`.

Categorical features included Cellularity, Chemotherapy, ER Status, HER2 Status, Hormone Therapy, Tumor Stage, Radio Therapy, and Oncotree Code. Missing feature values were replaced with `Unknown`, and rows with missing target labels were removed.

The data was split into training and test sets using an 80/20 split, with stratification and a fixed random state. The custom ID3 classifier and Scikit-learn's decision tree classifier were trained using a maximum depth of 3. The Scikit-learn model used entropy as its splitting criterion and one-hot encoding for the categorical features.

The models were compared using accuracy, precision, recall, F1-score, and confusion matrices.

### Comparison of Custom ID3 and Scikit-learn Decision Tree

The custom ID3 algorithm achieved an accuracy of 57.93%, while the Scikit-learn Decision Tree achieved an accuracy of 60.20%. Therefore, the Scikit-learn model performed slightly better overall on the test set.

The difference may be due to how the two models handle categorical features and choose the best splits. Scikit-learn also uses one-hot encoding, which may help it make better predictions for this dataset.

However, the results differ between the two classes. For the alive class, the custom ID3 model achieved a higher recall (0.47 compared with 0.36) and a higher F1-score (0.49 compared with 0.43). This indicates that the custom ID3 model identified a larger proportion of patients who were actually alive.

For the died class, the Scikit-learn model achieved a higher recall (0.78 compared with 0.66) and a higher F1-score (0.69 compared with 0.64). This suggests that it was more effective at identifying patients who had died.

### Confusion Matrix Comparison

The confusion matrices show that the custom ID3 model correctly classified 230 out of 397 test samples, whereas the Scikit-learn Decision Tree correctly classified 239 samples. This is consistent with their respective accuracies of 57.93% and 60.20%.

The custom ID3 model correctly identified 79 patients in the `alive` class and 151 in the `died` class. In comparison, the Scikit-learn model correctly identified 60 patients in the `alive` class and 179 in the `died` class.

Therefore, the Scikit-learn model was more effective at identifying patients in the `died` class, while the custom ID3 model correctly identified more patients in the `alive` class.
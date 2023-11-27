# Telco Customer Churn Prediction

This project addresses a classification machine learning task focused on predicting customer churn in a telecommunications company. Churn is identified based on whether customers left within the last month, labeled as 'yes' or 'no.'

The dataset utilized for this project is sourced from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn). It encompasses information such as:

- Customers who left within the last month (Churn column)
- Services subscribed by each customer, including phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account details such as tenure, contract, payment method, paperless billing, monthly charges, and total charges.
- Demographic information like gender, age range, and the presence of partners and dependents.

## Methodology
Initially, 20% of the data was reserved for final testing, stratified by the 'Churn' (target) column.

## Data Cleaning
- Converted the 'TotalCharges' column from object type to float type using `pd.to_numeric()` with errors parameter set to 'coerce' to parse invalid data to NaN.
- Imputed eight missing values in the 'TotalCharges' column with the mean() value.
- Verified absence of duplicates in the data.

## Exploratory Data Analysis
1. A count plot highlighted an imbalance in the distribution of churn rates.
2. Categorical feature count plots provided insights, such as even gender distribution and redundant information in categories like 'No Internet Service' and 'No Phone Service,' which were replaced with 'No.'
3. Histograms and box plots of continuous features revealed no outliers and a right-skewed 'TotalCharges' feature.
4. A scatter plot of 'MonthlyCharges' vs. 'TotalCharges' indicated a positive correlation, affecting the churn rate positively.

## Feature Encoding
Tested various encoding techniques, and One-Hot encoding on all categorical features yielded the best results.

## Feature Engineering
Binned the 'tenure' feature into six ranges to enhance its interpretability.

## Feature Scaling
Applied log transformation, specifically `np.log1p()`, to 'MonthlyCharges' and 'TotalCharges' due to their skewed distribution, proving superior to MinMaxScaler() and StandardScaler().

## Data Imbalance
Addressed potential class imbalance using the SMOTE (Synthetic Minority Oversampling Technique) library to synthetically increase the minority class ('yes').

## Preprocessing Function
Developed a Python function, `test_prep(dataframe)`, to combine and apply all previous preprocessing steps to test data, handling missing values based on the mean value in the training set.

## Models Training
Four models were trained and evaluated, with results reported using confusion matrices and classification reports:

1. Logistic Regression with best parameters: C=200, max_iter=1000.
2. Support Vector Classifier with best parameters: kernel='linear', C=20.
3. XGBoost Classifier with hyperparameters tuned using RandomizedSearchCV and StratifiedKFold.
4. Multi-layer Perceptron (MLP) Classifier.

```python
# Sample code for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Assuming 'X' is the feature matrix and 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Instantiate the logistic regression model with best parameters
logreg_model = LogisticRegression(C=200, max_iter=1000)

# Fit the model
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred = logreg_model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
```

Repeat similar code structure for other models mentioned in the project.

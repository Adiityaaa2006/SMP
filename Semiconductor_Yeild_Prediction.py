# 🗂️ ***Capstone Project - 2***

# 1. Import and Explore the data

## 1.1 Loading the data
"""
Loading the dataset `signal-data.csv` into a DataFrame and display basic information such as the first few rows, column names, and data types to understand its structure and content.

**Reasoning**:
Import pandas, load the CSV file into a DataFrame, display the first few rows, and show column information and data types.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the file is in the default Colab content directory or has been uploaded there
df_signal = pd.read_csv('signal-data.csv')
display(df_signal.head())
display(df_signal.info())

## 1.2 Explore data

"""
Perform exploratory data analysis (EDA) to understand the relationships and patterns within the signal data. This might involve visualizations or statistical tests depending on the nature of the data.

**Reasoning**:
Calculate and display descriptive statistics for numerical columns, count the distribution of 'Pass/Fail', and calculate correlations with 'Pass/Fail' to perform exploratory data analysis.
"""

# 1. Calculate and display descriptive statistics for numerical columns
display("Descriptive Statistics for Numerical Columns:")
display(df_signal.describe())

# 2. Analyze the distribution of the target variable 'Pass/Fail'
display("\nDistribution of 'Pass/Fail' column:")
display(df_signal['Pass/Fail'].value_counts())

# 2. Data Cleaning
# Data Cleaning is a process where identifying and correcting the errors in the data. The goal of Data cleaning is to ensure that data is accurate, complete, and consistent which is essential for Data Analysis.

## 2.1 Checking for Missing Values
# Clean the data by handling missing values, if any, and converting data types as necessary to prepare it for analysis and modeling.

missing_values = df_signal.isnull().sum()
print("Missing values per column:")
print(missing_values)

## 2.2 Data Cleaning Steps
# Following are the steps for data cleaning

# Step 1: Convert 'Time' column to datetime
df_signal['Time'] = pd.to_datetime(df_signal['Time'], errors='coerce')

# Step 2: Drop columns with more than 50% missing values
threshold = len(df_signal) * 0.5
df_signal = df_signal.dropna(thresh=threshold, axis=1)

# Step 3: Fill remaining NaNs using forward fill, then backward fill
df_signal = df_signal.fillna(method='ffill').fillna(method='bfill')

# Step 4: Remove duplicate rows
df_signal = df_signal.drop_duplicates()

# Step 5: Drop constant columns (same value in all rows)
df_signal = df_signal.loc[:, df_signal.nunique(dropna=False) > 1]

# Optional: Check the cleaned dataset
print(df_signal.info())
print(df_signal.head())

# Optional: Save the cleaned dataset
df_signal.to_csv("cleaned_signal_data.csv", index=False)

"""
 **step 1:** Loads the raw CSV data into a DataFrame named df_signal.

 **step 2:** Removes columns with more than 50% missing (NaN) values.

 **step 3:** Fills missing values using forward-fill first, then back-fill for any remaining.

 **step 4:**  Deletes any completely duplicate rows in the dataset.

 **step 5:** Removes columns where all values are the same

 **step 6:** Displays structure and non-null counts of the cleaned DataFrame.

 **step 7:** Cleaned dataset is modified in the existing data set.[adarshini raju]
"""

# Importing the Newly created dataset which has cleaned attributes and features.

df_new = pd.read_csv('cleaned_signal_data.csv')
display(df_new.head())
display(df_new.info())

# 3. Data Analysis and Visualization"""

display(df_signal.describe())

"""##Visualizing the distribution"""

sns.countplot(x='Pass/Fail', data=df_signal)
plt.title('Distribution of Pass/Fail')
plt.show()

sns.countplot(x='Time', data=df_signal)
plt.title('Time')
plt.show()

"""## Boxplot"""

sns.boxplot(data=df_signal[['1', '2', '3','4','6','7','8','9','10']])  # using start data
plt.title('Boxplot for Few starting signal')
plt.show()

"""
**observations**

- Have high median values

1.   Categories 1 & 2:
   - Have high median values
   -Have high median values
   -Show multiple outliers
2.  Category 3:
    - spread is wider
    -  Contains a large number of outliers, hinting at erratic behavior
3.  Categories 4 to 10:
    - Medians hover close to 0
    -  indicates almost no signal presence
    - Few outliers

"""

sns.boxplot(data=df_signal[['582', '583', '584','585','586','587']])  # using end data
plt.title('Boxplot for ending signals')
plt.show()

## Correlations

numerical_cols = df_signal.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df_signal[numerical_cols].corr()
display("\nCorrelations with 'Pass/Fail':")
display(correlation_matrix['Pass/Fail'].sort_values(ascending=False))

"""
1.   Categories 582 to 584, 586, 587:
     - Data is heavily concentrated near zero
     - Suggests little to no signal activity in these categories
    - likely just baseline readings
2.    Category 585:
     - Distinct outlier pushing up around 100
    - The rest of the data stays close to zero, similar to other categories

**We are excluding the Time coloumn as it is not numerical**
"""

## Correlation Map

plt.figure(figsize=(12, 8))
subset = df_signal.iloc[:, 1:10]  # using only first 10 data
sns.heatmap(subset.corr(), annot=True, cmap='PRGn')
plt.title('Correlation Between First 10 Signals')
plt.show()

# Lots of entries are near 0, like -0.0076, 0.0048, and -0.011, which typically signal no meaningful correlation.

plt.figure(figsize=(12, 8))
subset = df_signal.iloc[:, 1:10]  # First 10 signals only to avoid overload
sns.heatmap(subset.corr(), annot=False, cmap='spring')
plt.title('Correlation Between First 10 Signals')
plt.show()

##Pairploting

sns.pairplot(df_signal[['1', '2', '3','4']])
plt.suptitle('Pairplot of Selected Signals', y=1)
plt.show()

"""
# 4. Data Pre-processing

In this Section, We are going to preprocess to use the data in data correctly. Firstly, selecting the columns(attributes) for spliting the data into two part of X and Y. Where, X is for training data which is used for traning the model and Y is used as testing data which is used to test the models. Here we will use columns from 1 to 589 all columns excluding Time columns. For target we are going to use Pass/Fail columns which show the result.

## 4.1 Droping Irrelevant Columns

Here, we are removing the `Time` column from the data b'cuz it doesn't work from the dataset. So simply we will remove the column
"""

df_new.drop(["Time"], axis=1, inplace=True)
df_new.info()

"""
## 4.2 Spliting Data and Target

Here, selecting the all columns also all columns are in numerical values, none of the columns have string datatype data.
"""

X = df_new.drop(["Pass/Fail"], axis=1)
X.head()

Y = df_new["Pass/Fail"]
Y.head()

"""
## 4.3 Preparation of Test and Train Data

The final process where the data is seperate into for training and test. For this, we will be using `train_test_split` by importing it from `Sckikit-Learn Library`. We will be using 20% data for testing and 80% of data for traning.
"""

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=42)

xtrain.shape

xtrain.head()

xtest.head()

ytrain.head()

ytest.head()

"""
## 4.4 Normalization of Data

The values of the data may be so far from each other. This can sometimes lead to undesirable situations in regression algorithms. Therefore, we need to normalize the data
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

for column in xtrain.columns:
    xtrain[column] = scaler.fit_transform(xtrain[[column]])
    xtest[column] = scaler.transform(xtest[[column]])

xtrain.head()

"""
✅ Now the data is perfect for testing and traning of models.

# 5. Model Training, testing and tuning

This section involves training and evaluating machine learning models to classify signal data as Pass or Fail. We apply supervised algorithms with techniques like cross-validation, hyperparameter tuning, standardization, and class balancing to improve model performance. Models are assessed using classification reports and accuracy metrics, with the final selection based on their ability to handle both majority and minority classes effectively.

###5.1 Model training and testing for three models

We trained and evaluated three machine learning models — Random Forest, SVM, and Naive Bayes — using cross-validation and hyperparameter tuning. To handle class imbalance, we used class weighting and threshold adjustment. Each model's performance was compared based on accuracy and its ability to detect the minority class, helping us identify the most effective model for our classification task.

**First Model - RandomForestClassifier**

This section trains a Random Forest Classifier using RandomizedSearchCV to tune hyperparameters for better accuracy. The model uses class_weight='balanced' to address class imbalance. After training, predictions are made on the test set both with and without threshold adjustment (to improve class 1 detection). The model’s performance is evaluated using accuracy and detailed classification reports.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define a smaller hyperparameter grid
param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Instantiate base model with class_weight
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# RandomizedSearchCV
rand_search = RandomizedSearchCV(
    rf, param_distributions=param_grid_rf,
    n_iter=10, cv=5, scoring='accuracy', random_state=42,
    n_jobs=-1, verbose=1
)

# Fit model
rand_search.fit(xtrain, ytrain)
best_rf = rand_search.best_estimator_

# Predictions
y_pred_rf = best_rf.predict(xtest)

# Threshold adjustment
y_pred_proba = best_rf.predict_proba(xtest)
y_pred_custom = ((y_pred_proba[:, 1] >= 0.3).astype(int) * 2) - 1  # Convert 0 to -1, 1 stays 1

# Evaluation
print("Best Random Forest Parameters:", rand_search.best_params_)
print("Classification Report (Thresholded):\n", classification_report(ytest, y_pred_custom, zero_division=0))
print("Classification Report (Raw):\n", classification_report(ytest, y_pred_rf, zero_division=0))
print("Train Accuracy:", best_rf.score(xtrain, ytrain))
print("Test Accuracy:", accuracy_score(ytest, y_pred_rf))

"""
**Second Model - SVM (Support vector Machines)**

This section trains a Support Vector Machine (SVM) model using a pipeline that includes feature standardization (StandardScaler) and hyperparameter tuning via RandomizedSearchCV. The SVM is configured with class_weight='balanced' to handle class imbalance. The best model is selected based on cross-validated accuracy. Predictions are made on the test set, with and without threshold adjustment, and evaluated using classification reports and accuracy scores.
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Create a pipeline to scale and train
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', probability=True))
])

# Hyperparameters to tune
param_grid_svm = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}

# RandomizedSearchCV
svm_search = RandomizedSearchCV(
    pipeline_svm,
    param_distributions=param_grid_svm,
    n_iter=10,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit model
svm_search.fit(xtrain, ytrain)
best_svm = svm_search.best_estimator_

# Predict
y_pred_svm = best_svm.predict(xtest)
y_pred_proba_svm = best_svm.predict_proba(xtest)
y_pred_svm_thresh = ((y_pred_proba_svm[:, 1] >= 0.3).astype(int) * 2) - 1  # Thresholded to match -1/1

# Evaluation
print("Best SVM Parameters:", svm_search.best_params_)
print("Classification Report (Thresholded):\n", classification_report(ytest, y_pred_svm_thresh, zero_division=0))
print("Classification Report (Raw):\n", classification_report(ytest, y_pred_svm, zero_division=0))
print("Train Accuracy:", best_svm.score(xtrain, ytrain))
print("Test Accuracy:", accuracy_score(ytest, y_pred_svm))

"""
**Third Model - Naive Bayes**

This section trains a Gaussian Naive Bayes classifier using a pipeline that includes feature standardization with StandardScaler. Unlike the previous models, Naive Bayes does not support hyperparameter tuning or class weighting, so it serves as a fast and simple baseline model. Predictions are generated both with and without threshold adjustment to assess performance on the minority class. The model is evaluated using classification reports and accuracy metrics.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Naive Bayes pipeline
pipeline_nb = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# Fit the model
pipeline_nb.fit(xtrain, ytrain)

# Predict
y_pred_nb = pipeline_nb.predict(xtest)
y_pred_proba_nb = pipeline_nb.predict_proba(xtest)
y_pred_nb_thresh = ((y_pred_proba_nb[:, 1] >= 0.3).astype(int) * 2) - 1  # threshold adjustment

# Evaluation
print("Classification Report (Thresholded):\n", classification_report(ytest, y_pred_nb_thresh, zero_division=0))
print("Classification Report (Raw):\n", classification_report(ytest, y_pred_nb, zero_division=0))
print("Train Accuracy:", pipeline_nb.score(xtrain, ytrain))
print("Test Accuracy:", accuracy_score(ytest, y_pred_nb))

"""### 5.2 Comparing the three models to find the best trained model"""

import pandas as pd

# Create a dictionary of results
model_results = {
    'Model': ['Random Forest', 'SVM', 'Naive Bayes'],
    'Train Accuracy': [1.00, 1.00, 0.1963],
    'Test Accuracy': [0.9331, 0.9363, 0.2038],
    'F1-score (Class 1)': [0.00, 0.09, 0.14]
}

# Convert to DataFrame
results_df = pd.DataFrame(model_results)

# Display
print(" Model Comparison:")
display(results_df)

"""#### Comparing the models and selecting the best model

| Model             | Train Accuracy | Test Accuracy | F1-score (Class 1) | Observations                                                                     |
| ----------------- | -------------- | ------------- | ------------------ | -------------------------------------------------------------------------------- |
| **Random Forest** | 1.00           | 93.3%         | **0.00**           | Completely failed to predict class 1, despite high accuracy.                   |
| **SVM**           | 1.00           | 93.6%         | **0.09**           | Slight improvement: predicted a few class 1, but recall is still poor.         |
| **Naive Bayes**   | 19.6%          | 20.4%         | **0.14**           | Overpredicted class 1, but at the cost of huge drop in accuracy. Not reliable. |

The table compares three models based on their training accuracy, test accuracy, and F1-score for the minority class (class 1).

Random Forest achieved the highest overall accuracy but completely failed to predict any class 1 instances, resulting in an F1-score of 0 for the minority class.

SVM showed a slight improvement by correctly identifying a few class 1 samples, with a modest F1-score of 0.09, making it more balanced than Random Forest while maintaining similar accuracy.

Naive Bayes, on the other hand, predicted class 1 more aggressively, leading to a higher F1-score for class 1 (0.14), but it performed poorly overall, with very low accuracy, making it unsuitable for deployment.

Based on this comparison, SVM is the most balanced choice, offering a trade-off between class 1 detection and overall model reliability.

**Saving the selected model for future use**
"""

import joblib

# Save best SVM model
joblib.dump(best_svm, 'final_svm_model.pkl')

"""# 6. Conclusion

In this project, I have aimed to classify signal data into "Pass" or "Fail" categories using supervised machine learning techniques. The initial stages involved loading and exploring the dataset, followed by comprehensive data cleaning — including handling missing values, removing constant and duplicate columns, and converting time-based data. Exploratory Data Analysis (EDA) was performed to understand the distribution, identify correlations, and detect outliers, which informed our preprocessing decisions.

The cleaned and balanced dataset was then used for model training and evaluation. I experimented with three machine learning models: Random Forest, Support Vector Machine (SVM), and Naive Bayes. To ensure robust performance, I have used cross-validation, hyperparameter tuning, class weighting, and threshold adjustment.

Among the three models, SVM demonstrated the most balanced performance. While Random Forest achieved high accuracy, it failed to detect the minority class (class 1). Naive Bayes, though capable of detecting some class 1 samples, suffered from very poor overall accuracy. SVM provided a better trade-off, correctly identifying some minority class instances while maintaining strong overall accuracy.

In conclusion, SVM was selected as the final model, offering the best balance between class detection and generalization. The model was saved for future deployment, and this pipeline now serves as a strong foundation for further improvement, such as exploring advanced ensemble techniques or incorporating more sophisticated imbalance-handling strategies.
"""
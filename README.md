# nplearn
## Introduction
nplearn is a lightweight machine learning library where all core algorithms (such as linear regression, preprocessing, and evaluation metrics) are implemented using NumPy, following a Scikit-learn style API design. It is an ideal tool for understanding the fundamental principles of ML.
## Setup
This project is a local library. Please run it from the project root directory (the directory containing setup.py):
``` Python
# Install in editable mode
pip install -e .
```
Core dependency: numpy.
## Quick Start
Using a simple linear regression example to show how to use nplearn.
### 1. Import and Data Preparation
```Python
import numpy as np
from nplearn.linear_model import LinearRegression
from nplearn.model_selection import train_test_split

# Prepare sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2, 4, 5, 4, 5, 6, 7, 8])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
### 2. Training the Model and Making Predictions
```Python
# Instantiate the model
model = LinearRegression()

# Fit (train) model
model.fit(X_train, y_train)

# Print parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Make a prediction
y_pred = model.predict(X_test)
print(f"y_test: {y_test}")
print(f"y_pred: {y_pred}")
```
### 3. Model Evaluation
```Python
# Calculate R^2 Score
r2 = model.score(X_test, y_test)
print(f"R^2 Score: {r2}")
```

## Module Reference
|Model Name|Core Classes/Functions|Description|
|----------|----------|------------|
|linear_model|LinearRegression, RidgeRegression, LassoRegression, ElasticNet|Various forms of linear regression models.|
|preprocessing|StandardScaler, MinMaxScaler|Preprocessing class for feature scaling.|
|model_selection|train_test_split, KFold|Tools for data splitting and cross-validation.|
|metrics|mean_squared_error, r2_score|Common evaluation metrics, mainly used for regression tasks.|
|pipeline|Pipeline|Chain multiple steps together, such as scaling and modeling.|

## ⚖️ License
[MIT License](LICENSE)
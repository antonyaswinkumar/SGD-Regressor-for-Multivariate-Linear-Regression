# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import Libraries: 
Import required Python libraries: numpy, pandas, sklearn.

2. Load Dataset: 
Load or create a dataset with features (e.g., area, number of rooms, location index).
Include target variables (house price, number of occupants).

3. Preprocess Data: 
Handle missing values if present.
Scale features using StandardScaler to speed up convergence.

4. Split Dataset: 
Use train_test_split to split data into training and testing sets.

5. Train Model with SGD Regressor: 
Initialize SGDRegressor with a learning rate and max iterations.
Use MultiOutputRegressor for predicting multiple outputs.
Train model using the training set.

6. Make Predictions: 
Predict house price and number of occupants for the test set.

7. Evaluate Model: 
Use metrics such as r2_score and mean_squared_error to check performance.

8. Display Results: 
Print actual vs predicted values for better understanding.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ANTONY ASWIN KUMAR L 
RegisterNumber: 212225040024

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Features : no. of rooms ,area in (sq.ft), price
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
])
y = np.array([50, 45, 70, 80, 95])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SGD Regressor
sgd_reg = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_reg.fit(X_scaled, y)

# Coefficients and intercept
print("Weights (coefficients):", sgd_reg.coef_)
print("Intercept:", sgd_reg.intercept_)

# Predictions
y_pred = sgd_reg.predict(X_scaled)
print("Actual Price: ",y)
print("Predicted values: ", y_pred)

```

## Output:

<img width="870" height="96" alt="Screenshot 2026-05-14 110013" src="https://github.com/user-attachments/assets/51b6e2a8-c857-4b16-b976-1265d7506518" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

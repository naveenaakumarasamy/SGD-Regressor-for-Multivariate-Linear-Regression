# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import necessary libraries, including `numpy`, `pandas`, `matplotlib`, `SGDRegressor`, `MultiOutputRegressor`, `train_test_split`, `mean_squared_error`, and `StandardScaler`.

2. **Load Dataset**: Use `fetch_california_housing()` to load the California Housing dataset.

3. **Select Features and Target Variables**: Extract the first three features (`X`) and define the target (`Y`) by stacking the house price (`data.target`) and number of occupants (feature at index 6).

4. **Split Data into Training and Testing Sets**: Split the input data (`X`) and output data (`Y`) into training and testing sets using `train_test_split()`.

5. **Scale the Data**: Use `StandardScaler` to scale both input features (`X_train` and `X_test`) and the target variables (`Y_train` and `Y_test`) for better performance of the gradient-based optimization algorithm.

6. **Initialize SGD Regressor**: Create an instance of `SGDRegressor` with parameters such as `max_iter` for maximum iterations and `tol` as tolerance.

7. **MultiOutput Regressor**: Wrap the `SGDRegressor` model using `MultiOutputRegressor` to handle multiple output variables simultaneously.

8. **Train the Model**: Fit the model to the scaled training data (`X_train` and `Y_train`).

9. **Predict on Test Data**: Use the trained model to predict on the scaled test set (`X_test`).

10. **Inverse Transform and Evaluate**: Inverse-transform the predictions and actual values to their original scale. Calculate the **Mean Squared Error (MSE)** between the predicted and actual values.

11. **Display Results**: Print the Mean Squared Error and the first 5 predicted values for house prices and the number of occupants.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Naveenaa A K
RegisterNumber:  212222230094
*/
```

```PYTHON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data[:, :3]
Y = np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter = 1000, tol = 1e-3)

multi_output_regressor = MultiOutputRegressor(sgd)
multi_output_regressor.fit(X_train,Y_train)

#predict on test data
Y_pred = multi_output_regressor.predict(X_test)

Y_test = scaler_Y.inverse_transform(Y_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)

mse = mean_squared_error(Y_test,Y_pred)
print(f"Mean Squared Error: {mse}")

print("\nPredicted Values:",Y_pred[:5])
```

## Output:
![Screenshot 2024-09-05 215002](https://github.com/user-attachments/assets/dae860d2-b4c3-4619-8c09-43a6c00e3d7e)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

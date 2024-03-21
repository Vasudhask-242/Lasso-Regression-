# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:31:58 2024

@author: vaish
"""

#Problem Statement
'''
    An online car sales platform would like to improve its customer base and 
    their experience by providing them an easy way to buy and sell cars. For 
    this, they would like to have an automated model which can predict the price 
    of the car once the user inputs the required factors. Help the business 
    achieve the objective by applying Lasso and Ridge Regression on it. 
    Please use the below columns for the analysis: Price, Age_08_04, 
'''

# Business Objective
'''
    The business objective is to create an automated model that predicts the 
    price of cars on an online sales platform. By providing users with accurate 
    price predictions based on relevant factors, the platform aims to enhance customer 
    experience and facilitate buying and selling transactions.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/0-Assignments/Assignments/LassoReg/ToyotaCorolla (1).csv",encoding='ISO-8859-1')

data.shape
#(1436, 38)

summary = data.describe()


# Selecting relevant columns for analysis
selected_columns = ['Price', 'Age_08_04']
data = data[selected_columns]

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Data preprocessing
X = data.drop(columns=['Price'])
y = data['Price']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model building
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # Adjust alpha as needed
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)
    return rmse, r_squared

# Evaluate Linear Regression model
lr_rmse, lr_r_squared = evaluate_model(lr_model, X_test_scaled, y_test)

# Evaluate Lasso Regression model
lasso_rmse, lasso_r_squared = evaluate_model(lasso_model, X_test_scaled, y_test)

# Evaluate Ridge Regression model
ridge_rmse, ridge_r_squared = evaluate_model(ridge_model, X_test_scaled, y_test)

# Print the results
print("Linear Regression RMSE:", lr_rmse)
#1800.934037639833
print("Linear Regression R-Squared:", lr_r_squared)
#0.7569201277242213
print("Lasso Regression RMSE:", lasso_rmse)
#1800.9421213521496
print("Lasso Regression R-Squared:", lasso_r_squared)
#0.7569179455319518
print("Ridge Regression RMSE:", ridge_rmse)
#1801.1594530035989
print("Ridge Regression R-Squared:", ridge_r_squared)
#0.7568592733388284

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_pred, label='Linear Regression')
plt.scatter(y_test, lasso_pred, label='Lasso Regression')
plt.scatter(y_test, ridge_pred, label='Ridge Regression')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Price')
plt.legend()
plt.show()

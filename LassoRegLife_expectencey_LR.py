# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:40:21 2024

@author: HP
"""

#problem statement
'''
    Data of various countries and the factors affecting their life expectancy 
    has been recorded over the past few decades. An analytics firm would like 
    to know how it varies country wise and what factors are influential. 
    Use your skills to analyze the data and build a Lasso and Ridge Regression 
    model and summarize the output. Snapshot of the dataset is given below.
'''

#Business objective
'''
    The objective is to analyze the factors affecting life expectancy across 
    various countries and build predictive models using Lasso and Ridge 
    Regression techniques. The goal is to gain insights into how life 
    expectancy varies by country and identify influential factors.
'''

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:/0-Assignments/Assignments/LassoReg/Life_expectencey_LR.csv")

# Data preprocessing
# Handling missing values (if any)
print("Missing values:\n", data.isnull().sum())

# Drop rows with missing values for simplicity
data.dropna(inplace=True)

# Selecting relevant columns for analysis
selected_columns = ['Life_expectancy', 'Adult_Mortality']  # Assuming these are the factors affecting life expectancy
data = data[selected_columns]

# Splitting data into features and target variable
X = data.drop(columns=['Life_expectancy'])
y = data['Life_expectancy']

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
#6.1572874826237465

print("Linear Regression R-Squared:", lr_r_squared)
#0.46619455204043514

print("Lasso Regression RMSE:", lasso_rmse)
#6.162182434814311

print("Lasso Regression R-Squared:", lasso_r_squared)
#0.4653454798529514

print("Ridge Regression RMSE:", ridge_rmse)
#6.157491603397213

print("Ridge Regression R-Squared:", ridge_r_squared)
#0.46615915899201565

import matplotlib.pyplot as plt

# Plot actual vs. predicted values for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_pred, color='blue', label='Linear Regression')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs. Predicted Life Expectancy (Linear Regression)')
plt.legend()
plt.show()

# Plot actual vs. predicted values for Lasso Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lasso_pred, color='red', label='Lasso Regression')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs. Predicted Life Expectancy (Lasso Regression)')
plt.legend()
plt.show()

# Plot actual vs. predicted values for Ridge Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_pred, color='green', label='Ridge Regression')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs. Predicted Life Expectancy (Ridge Regression)')
plt.legend()
plt.show()

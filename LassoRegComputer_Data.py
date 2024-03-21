# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:49:38 2024

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:/0-Assignments/Assignments/LassoReg/Computer_Data (1).csv")
data.shape
# (6259, 11)

data.describe()

# Step 1: Data preprocessing
# Assuming there are no missing values in the dataset
# No further preprocessing needed based on the provided description

# Step 2: Feature scaling and encoding categorical variables
# Separate features and target variable
X = data.drop(columns=['Unnamed: 0', 'price'])  # Exclude irrelevant columns
y = data['price']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust alpha as needed
lasso_model.fit(X_train, y_train)

# Step 5: Evaluate the model
lasso_pred = lasso_model.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r_squared = r2_score(y_test, lasso_pred)

print("Lasso Regression RMSE:", lasso_rmse)
# 283.41608112808245

print("Lasso Regression R-Squared:", lasso_r_squared)
#0.7541138174533601
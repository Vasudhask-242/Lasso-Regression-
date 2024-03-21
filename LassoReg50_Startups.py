# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:22:07 2024

@author: HP
"""

#Problem Statements:
'''
    Officeworks is a leading retail store in Australia, with numerous outlets 
    around the country. The manager would like to improve the customer experience 
    by providing them online predictive prices for their laptops if they want to 
    sell them. To improve this experience the manager would like us to build a 
    model which is sustainable and accurate enough. Apply Lasso and Ridge Regression 
    model on the dataset and predict the price, given other attributes. 
    Tabulate R squared, RMSE, and correlation values.
'''

#Business Objective:
'''
    The objective is to improve customer experience at Officeworks, a leading 
    retail store in Australia, by providing online predictive prices for 
    laptops that customers want to sell. This involves building a sustainable 
    and accurate predictive model.
'''
# Constraints:
'''    
The model should be accurate enough to provide reliable price predictions.
The model should be sustainable for long-term use in the business operations of Officeworks
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/0-Assignments/Assignments/LassoReg/50_Startups (1).csv")

data.shape
#(50, 5)

data.columns
'''
Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')
'''

data.describe()
'''
           R&D Spend  Administration  Marketing Spend         Profit
count      50.000000       50.000000        50.000000      50.000000
mean    73721.615600   121344.639600    211025.097800  112012.639200
std     45902.256482    28017.802755    122290.310726   40306.180338
min         0.000000    51283.140000         0.000000   14681.400000
25%     39936.370000   103730.875000    129300.132500   90138.902500
50%     73051.080000   122699.795000    212716.240000  107978.190000
75%    101602.800000   144842.180000    299469.085000  139765.977500
max    165349.200000   182645.560000    471784.100000  192261.830000
'''


# Data preprocessing
# Assuming no missing values and feature engineering for now

# Encoding categorical variable 'State'
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Splitting data into features and target variable
X = data.drop(columns=['Profit'])
y = data['Profit']

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
print("Linear Regression R-Squared:", lr_r_squared)
#9055.957323497809

print("Lasso Regression RMSE:", lasso_rmse)
#9055.853740793507

print("Lasso Regression R-Squared:", lasso_r_squared)
#0.8987289581629456

print("Ridge Regression RMSE:", ridge_rmse)
#9204.987745500264

print("Ridge Regression R-Squared:", ridge_r_squared)
#0.8953659807905756

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=lr_pred, label='Linear Regression')
sns.scatterplot(x=y_test, y=lasso_pred, label='Lasso Regression')
sns.scatterplot(x=y_test, y=ridge_pred, label='Ridge Regression')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Predicted vs Actual Profit')
plt.legend()
plt.show()

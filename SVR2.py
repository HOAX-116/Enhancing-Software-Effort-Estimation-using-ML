#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:01:43 2024

@author: cheera
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
data = pd.read_csv("/home/cheera/Documents/NITW /desharnais.csv")
target_var = "Effort"
def feat_sel(data, corr_threshold=0.5):
    corr_matrix = data.corr()
    selected_features = corr_matrix[target_var].sort_values(ascending=False)[
        lambda x: abs(x) > corr_threshold
    ].index.drop(target_var)
    return selected_features
selected_features = feat_sel(data, corr_threshold=0.5)
print("Selected features with correlation > 0.5:")
print(selected_features)
# Initialize the data
X = data[selected_features]
y = data[target_var]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Initial hyperparameters
C = 10
epsilon = 0.01
kernel = 'rbf'

# Store the MSE for each iteration
mse_values = []

# Function to create the contour plot
def plot_contour(X, y, model, iteration):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Iteration {iteration}')
    plt.show()

# Iterative optimization process
for i in range(100):
    # Initialize the SVR model with current hyperparameters
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)
    
    # Plot the contour plot for the current iteration
    plot_contour(X_test.to_numpy(), y_test.to_numpy(), model, i)
    
    print(f"Iteration: {i}")
    print(f"MSE: {mse}")
    
    # Adjust hyperparameters slightly for the next iteration
    # (For simplicity, adjust C and epsilon by small amounts)
    C = C * (1 - 0.01)
    epsilon = epsilon * (1 - 0.01)

# Plot the MSE values over iterations
plt.plot(range(100), mse_values)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE over Iterations')
plt.show()

mse_m=mse/100
print("Mean of all Mean Square Errors in each iteration:{mse_m}")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.to_string(index=False))
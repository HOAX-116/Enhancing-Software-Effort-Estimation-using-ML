#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:08:27 2024

@author: cheera
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ELM:


    def __init__(self, n_hidden_neurons, activation_function='sigmoid'):
        self.n_hidden_neurons = n_hidden_neurons
        self.activation_function = activation_function

    def _activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    def fit(self, X, y):
        self.input_weights = np.random.uniform(-1, 1, (X.shape[1], self.n_hidden_neurons))
        self.biases = np.random.uniform(-1, 1, (self.n_hidden_neurons,))
        H = self._activation(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self._activation(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)

# Load the dataset
data=pd.read_csv("/home/cheera/Documents/NITW /desharnais.csv") # Replace with your actual CSV file
features = data[['Transactions', 'Entities', 'PointsNonAdjust', 'PointsAjust']]
target = data['Effort']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=15)


# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the ELM model
elm = ELM(n_hidden_neurons=50, activation_function='sigmoid')
elm.fit(X_train, y_train)

# Predict and evaluate
y_pred_train = elm.predict(X_train)
y_pred_test = elm.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training Mean Squared Error: {mse_train}")
print(f"Testing Mean Squared Error: {mse_test}")
print(f"Training R^2 Score: {r2_train}")
print(f"Testing R^2 Score: {r2_test}")

# Plotting graphs
plt.figure(figsize=(14, 6))

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual Effort')
plt.ylabel('Predicted Effort')
plt.title('Training Data')

# Plot testing data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Effort')
plt.ylabel('Predicted Effort')
plt.title('Testing Data')

plt.tight_layout()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:01:42 2024

@author: cheera
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/home/cheera/Documents/NITW /desharnais.csv')

# Preprocess the dataset
features = data.iloc[:, :-1].values  # All columns except the last one
actual_effort = data.iloc[:, -1].values  # The last column

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Define the objective function
def analogy_based_estimation(params, features, actual_effort):
    """Objective function to minimize: Mean Absolute Error of analogy-based estimation."""
    weights = params[:len(features[0])]  # Weights for the features
    k = int(np.clip(params[-1], 1, len(features) - 1))  # Number of analogies to consider
    
    estimated_efforts = []
    
    for i in range(len(features)):
        # Calculate the distance between the current project and all other projects
        distances = np.linalg.norm((features - features[i]) * weights, axis=1)
        # Get the indices of the k closest projects
        closest_indices = distances.argsort()[1:k+1]  # Exclude the project itself (at index 0)
        if len(closest_indices) == 0:
            continue  # Skip if no neighbors are found
        # Calculate the estimated effort as the mean effort of the k closest projects
        estimated_effort = np.mean(actual_effort[closest_indices])
        estimated_efforts.append(estimated_effort)
    
    # Ensure estimated_efforts is not empty to avoid runtime errors
    if len(estimated_efforts) == 0:
        return np.inf  # Return a high error if no valid estimations are made
    
    return mean_absolute_error(actual_effort, estimated_efforts)

# Firefly Algorithm
def firefly_algorithm(objective_function, features, actual_effort, dim, n_fireflies=20, max_iter=100,
                      alpha=0.5, beta_min=0.2, gamma=1.0):
    """Firefly Algorithm to minimize the objective_function."""
    fireflies = np.random.uniform(0, 1, (n_fireflies, dim))
    light_intensity = np.apply_along_axis(objective_function, 1, fireflies, features, actual_effort)
    
    best_firefly = fireflies[np.argmin(light_intensity)]
    best_intensity = np.min(light_intensity)
    
    all_fireflies_positions = [fireflies.copy()]
    
    for iteration in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if light_intensity[j] < light_intensity[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta_min + (1 - beta_min) * np.exp(-gamma * r ** 2)
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + \
                                   alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] = np.clip(fireflies[i], 0, 1)
                    light_intensity[i] = objective_function(fireflies[i], features, actual_effort)
                    
                    if light_intensity[i] < best_intensity:
                        best_firefly = fireflies[i]
                        best_intensity = light_intensity[i]
        
        all_fireflies_positions.append(fireflies.copy())
        alpha *= 0.97
    
    return best_firefly, best_intensity, all_fireflies_positions

# Parameters
dim = features.shape[1] + 1  # Number of features + 1 for the k parameter
best_solution, best_value, all_fireflies_positions = firefly_algorithm(analogy_based_estimation, features, actual_effort, dim)

print("Best solution found:", best_solution)
print("Objective function value:", best_value)

# Calculate performance measures
def calculate_performance_measures(best_solution, features, actual_effort):
    weights = best_solution[:len(features[0])]
    k = int(np.clip(best_solution[-1], 1, len(features) - 1))
    
    estimated_efforts = []
    for i in range(len(features)):
        distances = np.linalg.norm((features - features[i]) * weights, axis=1)
        closest_indices = distances.argsort()[1:k+1]
        if len(closest_indices) == 0:
            continue
        estimated_effort = np.mean(actual_effort[closest_indices])
        estimated_efforts.append(estimated_effort)
    
    if len(estimated_efforts) == 0:
        return np.inf, np.inf, np.inf, np.inf  # Return high errors if no valid estimations are made
    
    mae = mean_absolute_error(actual_effort, estimated_efforts)
    mape = np.mean(np.abs((actual_effort - np.array(estimated_efforts)) / actual_effort)) * 100
    mmre = np.mean(np.abs((actual_effort - np.array(estimated_efforts)) / actual_effort))
    pred25 = np.mean(np.abs((actual_effort - np.array(estimated_efforts)) / actual_effort) < 0.25) * 100
    
    return mae, mape, mmre, pred25

mae, mape, mmre, pred25 = calculate_performance_measures(best_solution, features, actual_effort)

print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"MMRE: {mmre}")
print(f"PRED(25): {pred25}%")

# Plotting the firefly positions (2D visualization for simplicity)
def plot_firefly_positions(all_fireflies_positions):
    fig, ax = plt.subplots()
    for i, fireflies in enumerate(all_fireflies_positions):
        ax.clear()
        ax.scatter(fireflies[:, 0], fireflies[:, 1], c='blue', marker='o', label='Fireflies')
        ax.set_title(f'Iteration {i}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.pause(0.1)

plot_firefly_positions(all_fireflies_positions)
plt.show()

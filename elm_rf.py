import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("/home/cheera/Documents/NITW /desharnais.csv")
X = data.drop("Effort", axis=1)
y = data["Effort"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
elm = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4] 
}
grid_search = GridSearchCV(elm, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R^2): {r2:.4f}")
while True:
    print("\nDo you want to make predictions on new data? (y/n)")
    user_choice = input().lower()
    if user_choice == 'y':
        new_data = {}  
        for feature in X.columns:
            value = input(f"Enter value for '{feature}': ")
            new_data[feature] = value
        new_data_df = pd.DataFrame([new_data])
        new_data_scaled = scaler.transform(new_data_df)      
        new_predictions = best_model.predict(new_data_scaled)
        print(f"Prediction for the new data: {new_predictions[0]:.4f}")
    elif user_choice == 'n':
        print("Exiting prediction loop.")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
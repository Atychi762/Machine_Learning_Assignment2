import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('datasets/steel.csv')

# Feature scaling
scaler = StandardScaler()
feature_cols = dataset.columns.drop('tensile_strength')
dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

kf = KFold(n_splits=10, shuffle=False)
default_average_mean_squared_error = 0
tuned_average_mean_squared_error = 0

mse_output = []

# Training and evaluation using 10-Fold Cross-Validation
for train, test in kf.split(dataset):
    train_data = dataset.iloc[train]
    test_data = dataset.iloc[test]

    # Defining test and target sets
    X_train = train_data.drop('tensile_strength', axis=1)
    y_train = train_data['tensile_strength']
    X_test = test_data.drop('tensile_strength', axis=1)
    y_test = test_data['tensile_strength']

    # Training the default GBR model
    default_model = GradientBoostingRegressor()
    default_model.fit(X_train, y_train)

    default_predictions = default_model.predict(X_test)
    default_mse = mean_squared_error(y_test, default_predictions)

    # Training the tuned GBR model
    # Varied hyperparameters are learning rate and n_estimators
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25],
        "n_estimators": [100, 200, 300, 400, 500]
    }
    # Using GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        GradientBoostingRegressor(),
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    # Using the best hyperparameters found
    tuned_predictions = best_model.predict(X_test)
    tuned_mse = mean_squared_error(y_test, tuned_predictions)

    # Appending MSE for the current fold
    mse_output.append({"Default_MSE": default_mse,
                       "Tuned_MSE": tuned_mse})

    default_average_mean_squared_error += default_mse
    tuned_average_mean_squared_error += tuned_mse

# Appending average MSE to the results
default_average_mean_squared_error /= 10
tuned_average_mean_squared_error /= 10

mse_output.append({"Default_MSE": default_average_mean_squared_error,
                   "Tuned_MSE": tuned_average_mean_squared_error})

# Saving MSE results to a CSV file
mse_df = pd.DataFrame(mse_output)
mse_df.to_csv('datasets/Gradient_Boosting_MSE_Results.csv', index=False)

print("Gradient Boosting Regressor Training Complete. MSE results saved to 'datasets/Gradient_Boosting_MSE_Results.csv'.")
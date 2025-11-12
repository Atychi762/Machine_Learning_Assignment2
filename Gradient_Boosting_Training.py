import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('datasets/steel.csv')

# Feature scaling
scaler = StandardScaler()
feature_cols = dataset.columns.drop('tensile_strength')
dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

kf = KFold(n_splits=10, shuffle=False)
default_average_root_mean_squared_error = 0
tuned_average_root_mean_squared_error = 0

rmse_output = []

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
    default_rmse = root_mean_squared_error(y_test, default_predictions)

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
    tuned_rmse = root_mean_squared_error(y_test, tuned_predictions)

    # Appending RMSE for the current fold
    rmse_output.append({"Default_RMSE": default_rmse,
                       "Tuned_RMSE": tuned_rmse})

    default_average_root_mean_squared_error += default_rmse
    tuned_average_root_mean_squared_error += tuned_rmse

# Appending average RMSE to the results
default_average_root_mean_squared_error /= 10
tuned_average_root_mean_squared_error /= 10

rmse_output.append({"Default_RMSE": default_average_root_mean_squared_error,
                   "Tuned_RMSE": tuned_average_root_mean_squared_error})

# Saving RMSE results to a CSV file
rmse_df = pd.DataFrame(rmse_output)
rmse_df.to_csv('datasets/Gradient_Boosting_RMSE_Results.csv', index=False)

print("Gradient Boosting Regressor Training Complete. RMSE results saved to 'datasets/Gradient_Boosting_RMSE_Results.csv'.")
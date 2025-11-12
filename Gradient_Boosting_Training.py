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
default_average_rmse_test = 0
tuned_average_rmse_test = 0
default_average_rmse_train = 0
tuned_average_rmse_train = 0

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

    # Calculating RMSE on training data
    default_train_predictions = default_model.predict(X_train)
    default_train_rmse = root_mean_squared_error(y_train, default_train_predictions)
    
    # Calculating RMSE on test data
    default_test_predictions = default_model.predict(X_test)
    default_test_rmse = root_mean_squared_error(y_test, default_test_predictions)

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
    # Calculating RMSE on training data
    tuned_train_predictions = best_model.predict(X_train)
    tuned_train_rmse = root_mean_squared_error(y_train, tuned_train_predictions)

    # Calculating RMSE on test data
    tuned_test_predictions = best_model.predict(X_test)
    tuned_test_rmse = root_mean_squared_error(y_test, tuned_test_predictions)

    # Appending RMSE for the current fold
    rmse_output.append({"Default_TrainData_RMSE": default_train_rmse,
                        "Default_TestData_RMSE": default_test_rmse,
                       "Tuned_TrainData_RMSE": tuned_train_rmse,
                       "Tuned_TestData_RMSE": tuned_test_rmse})

    default_average_rmse_test += default_test_rmse
    tuned_average_rmse_test += tuned_test_rmse
    default_average_rmse_train += default_train_rmse
    tuned_average_rmse_train += tuned_train_rmse

# Appending average RMSE to the results
default_average_rmse_test /= 10
tuned_average_rmse_test /= 10
default_average_rmse_train /= 10
tuned_average_rmse_train /= 10

rmse_output.append({ "Default_TrainData_RMSE": default_average_rmse_train,
                   "Default_TestData_RMSE": default_average_rmse_test,
                   "Tuned_TrainData_RMSE": tuned_average_rmse_train,
                   "Tuned_TestData_RMSE": tuned_average_rmse_test})

# Saving RMSE results to a CSV file
rmse_df = pd.DataFrame(rmse_output)
rmse_df.to_csv('datasets/Gradient_Boosting_RMSE_Results.csv', index=False)

print("Gradient Boosting Regressor Training Complete. RMSE results saved to 'datasets/Gradient_Boosting_RMSE_Results.csv'.")
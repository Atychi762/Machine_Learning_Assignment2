import pandas as pd
import numpy as np

from warnings import filterwarnings
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.exceptions import ConvergenceWarning

import Compute_Stats

filterwarnings("ignore", category=ConvergenceWarning)

dataset = pd.read_csv('datasets/steel.csv')

# Feature scaling
scaler = StandardScaler()
feature_cols = dataset.columns.drop('tensile_strength')
dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

kf = KFold(n_splits=10, shuffle=True, random_state=42)

default_average_rmse_test = 0
tuned_average_rmse_test = 0
default_average_rmse_train = 0
tuned_average_rmse_train = 0

default_strength_pred_and_res = []
tuned_strength_pred_and_res = []

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
    default_model = MLPRegressor()
    default_model.fit(X_train, y_train)

    # Calculating RMSE on training data
    default_train_predictions = default_model.predict(X_train)
    default_train_rmse = root_mean_squared_error(y_train, default_train_predictions)
    default_strength_pred_and_res.append((train, default_train_predictions - y_train.values))
    
    # Calculating RMSE on test data
    default_test_predictions = default_model.predict(X_test)
    default_test_rmse = root_mean_squared_error(y_test, default_test_predictions)
    default_strength_pred_and_res.append((test, default_test_predictions - y_test.values))

    # Feature selection on training data
    select_k_best = SelectKBest(score_func=f_regression, k=6)
    X_train_tuned = select_k_best.fit_transform(X_train, y_train)
    X_test_tuned = select_k_best.transform(X_test)

    # Training the tuned GBR model
    # Varied hyperparameters are learning rate init and max_iter
    param_grid = {
        "learning_rate_init": [0.00001, 0.0001,0.001, 0.01],
        "max_iter": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }
    # Using GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        MLPRegressor(),
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid_search.fit(X_train_tuned, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters for current fold: {grid_search.best_params_}")
    # Using the best hyperparameters found
    # Calculating RMSE on training data
    tuned_train_predictions = best_model.predict(X_train_tuned)
    tuned_train_rmse = root_mean_squared_error(y_train, tuned_train_predictions)
    tuned_strength_pred_and_res.append((train, tuned_train_predictions - y_train.values))

    # Calculating RMSE on test data
    tuned_test_predictions = best_model.predict(X_test_tuned)
    tuned_test_rmse = root_mean_squared_error(y_test, tuned_test_predictions)
    tuned_strength_pred_and_res.append((test, tuned_test_predictions - y_test.values))


    # Appending RMSE for the current fold
    rmse_output.append({"Learning_Rate_init": grid_search.best_params_['learning_rate_init'],
                        "max_iter": grid_search.best_params_['max_iter'],
                        "Default_TrainData_RMSE": round(default_train_rmse, 2),
                        "Default_TestData_RMSE": round(default_test_rmse, 2),
                        "Tuned_TrainData_RMSE": round(tuned_train_rmse, 2),
                        "Tuned_TestData_RMSE": round(tuned_test_rmse, 2)})

    default_average_rmse_test += default_test_rmse
    tuned_average_rmse_test += tuned_test_rmse
    default_average_rmse_train += default_train_rmse
    tuned_average_rmse_train += tuned_train_rmse


# Appending average RMSE to the results
default_average_rmse_test /= 10
tuned_average_rmse_test /= 10
default_average_rmse_train /= 10
tuned_average_rmse_train /= 10

rmse_output.append({"Learning_Rate_init": None,
                    "max_iter": None,
                    "Default_TrainData_RMSE": round(default_average_rmse_train, 2),
                    "Default_TestData_RMSE": round(default_average_rmse_test, 2),
                    "Tuned_TrainData_RMSE": round(tuned_average_rmse_train, 2),
                    "Tuned_TestData_RMSE": round(tuned_average_rmse_test, 2)})

# Saving RMSE results to a CSV file
rmse_df = pd.DataFrame(rmse_output)
rmse_df.to_csv("datasets/MLP_RMSE.csv", index=False)

n_samples = len(dataset)
col_names = []
for i in range(1, 11):
    col_names.append(f"fold_{i}_train")
    col_names.append(f"fold_{i}_test")

diffs_df = pd.DataFrame(index=range(n_samples), columns=col_names, dtype=float)

for idx, (row_indices, diffs) in enumerate(default_strength_pred_and_res):
    col = col_names[idx]
    diffs_df.loc[row_indices, col] = diffs

diffs_df.to_csv("datasets/Default_MLP_Strength_Differences.csv", index=False)

for idx, (row_indices, diffs) in enumerate(tuned_strength_pred_and_res):
    col = col_names[idx]
    diffs_df.loc[row_indices, col] = diffs

diffs_df.to_csv("datasets/Tuned_MLP_Strength_Differences.csv", index=False)

print("MLP Regressor Training Complete. RMSE results saved to 'datasets/MLP_RMSE.csv'.")

# Compute interpretable stats from the collected RMSE results
Compute_Stats.main("datasets/steel.csv", "datasets/MLP_RMSE.csv")

print("Computed interpretable statistics and saved to 'datasets/MLP_RMSE_PercentErrors.csv'.")
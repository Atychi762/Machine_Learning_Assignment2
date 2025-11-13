# Compute interpretable stats for tensile_strength and the RMSEs
import pandas as pd
import numpy as np

def main ():
    steel_dataset = pd.read_csv('datasets/steel.csv')
    rmse_dataset = pd.read_csv('datasets/Gradient_Boosting_RMSE_Results.csv')

    # Compute basic stats for tensile_strength
    tensile_strength_values = steel_dataset['tensile_strength']
    mean_tensile_strength = tensile_strength_values.mean()
    median_tensile_strength = tensile_strength_values.median()
    std_tensile_strength = tensile_strength_values.std()
    range_tensile_strength = tensile_strength_values.max() - tensile_strength_values.min()

    # RMSE column names
    default_test_col = 'Default_TestData_RMSE'
    tuned_test_col = 'Tuned_TestData_RMSE'
    default_train_col = 'Default_TrainData_RMSE'
    tuned_train_col = 'Tuned_TrainData_RMSE'


    # compute averages (for per-fold results we take the mean)
    avg_default_test_rmse = rmse_dataset[default_test_col].mean()
    avg_tuned_test_rmse = rmse_dataset[tuned_test_col].mean()
    avg_default_train_rmse = rmse_dataset[default_train_col].mean()
    avg_tuned_train_rmse = rmse_dataset[tuned_train_col].mean()

    print(f"mean(tensile_strength) = {mean_tensile_strength:.2f}")
    print(f"median = {median_tensile_strength:.2f}, std = {std_tensile_strength:.2f}, range = {range_tensile_strength:.2f}\n")

    # average RMSEs and their average percent errors
    print(f"Avg {default_test_col}  = {avg_default_test_rmse:.2f} -> {100*avg_default_test_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {tuned_test_col}    = {avg_tuned_test_rmse:.2f} -> {100*avg_tuned_test_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {default_train_col} = {avg_default_train_rmse:.2f} -> {100*avg_default_train_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {tuned_train_col}   = {avg_tuned_train_rmse:.2f} -> {100*avg_tuned_train_rmse/mean_tensile_strength:.2f}% average error\n")

    # per-fold percent errors for all four RMSE columns
    for col in (default_test_col, default_train_col, tuned_test_col, tuned_train_col):
        pct_col = f"{col}_Percent_Error"
        rmse_dataset[pct_col] = round(100 * rmse_dataset[col] / mean_tensile_strength, 2)

    pd.DataFrame(rmse_dataset).to_csv('datasets/Gradient_Boosting_RMSE_PercentErrors.csv', index=False)
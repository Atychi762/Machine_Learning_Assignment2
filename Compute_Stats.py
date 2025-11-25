# Compute interpretable stats for tensile_strength and the RMSEs
import pandas as pd
import numpy as np

def main (steel_path, rmse_path):
    steel_dataset = pd.read_csv(steel_path)
    rmse_dataset = pd.read_csv(rmse_path)

    # Compute the mean for tensile_strength
    mean_tensile_strength = steel_dataset["tensile_strength"].mean(skipna=True)

    # RMSE column names
    default_test_col = "Default_TestData_RMSE"
    tuned_test_col = "Tuned_TestData_RMSE"
    default_train_col = "Default_TrainData_RMSE"
    tuned_train_col = "Tuned_TrainData_RMSE"

    # compute averages (for per-fold results we take the mean)
    avg_default_test_rmse = rmse_dataset[default_test_col].mean(skipna=True)
    avg_tuned_test_rmse = rmse_dataset[tuned_test_col].mean(skipna=True)
    avg_default_train_rmse = rmse_dataset[default_train_col].mean(skipna=True)
    avg_tuned_train_rmse = rmse_dataset[tuned_train_col].mean(skipna=True)

    # average RMSEs and their average percent errors
    print(f"Avg {default_test_col}  = {avg_default_test_rmse:.2f} -> {100*avg_default_test_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {tuned_test_col}    = {avg_tuned_test_rmse:.2f} -> {100*avg_tuned_test_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {default_train_col} = {avg_default_train_rmse:.2f} -> {100*avg_default_train_rmse/mean_tensile_strength:.2f}% average error")
    print(f"Avg {tuned_train_col}   = {avg_tuned_train_rmse:.2f} -> {100*avg_tuned_train_rmse/mean_tensile_strength:.2f}% average error\n")

    # per-fold percent errors for all four RMSE columns
    for col in (default_test_col, default_train_col, tuned_test_col, tuned_train_col):
        pct_col = f"{col}_Percent_Error"
        rmse_dataset[pct_col] = round(100 * rmse_dataset[col] / mean_tensile_strength, 2)

    pd.DataFrame(rmse_dataset).to_csv(rmse_path.replace(".csv", "_PercentErrors.csv"), index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: refactor to avoid code duplication with GBR plotting script

rmse_data = pd.read_csv("datasets/MLP/MLP_RMSE_PercentErrors.csv")
rmse_data_no_avg = rmse_data[:-1] # Exclude the last row which contains averages
default_MLP_train_diff = pd.read_csv("datasets/MLP/Default_MLP_Strength_Differences.csv")
tuned_MLP_train_diff = pd.read_csv("datasets/MLP/Tuned_MLP_Strength_Differences.csv")

# Plotting RMSE for Default and Tuned models
plt.figure(figsize=(10, 6))
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Default_TestData_RMSE"], marker="o", label="Default Model Testing RMSE", color="blue")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Tuned_TestData_RMSE"], marker="o", label="Tuned Model Testing RMSE", color="orange")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Default_TrainData_RMSE"], marker="o", label="Default Model Training RMSE", color="green")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Tuned_TrainData_RMSE"], marker="o", label="Tuned Model Training RMSE", color="red")
plt.title("MLP Regressor Root Mean Squared Error Comparison")
plt.xlabel("Fold Index")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend()
plt.grid()
plt.savefig("images/MLP_RMSE_Comparison.png")


y_scale = list(range(0, 66, 10))
# Plotting Percent Error for Default and Tuned models
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("MLP Regressor Percent Error Comparison")
fig.subplots_adjust(hspace=0.4)  # add vertical space between subplots

# Top subplot - Default model
ax = axes[0]
ax.set_title("Percent Error for Default Model")
ax.bar(rmse_data_no_avg.index - 0.1, rmse_data_no_avg["Default_TrainData_RMSE_Percent_Error"], width=0.2, label="Default Model Training Data Percent Error", color="blue")
ax.bar(rmse_data_no_avg.index + 0.1, rmse_data_no_avg["Default_TestData_RMSE_Percent_Error"], width=0.2, label="Default Model Test Data Percent Error", color="orange")
ax.set_xlabel("Fold Index")
ax.set_ylabel("Percent Error (%)")
ax.set_yticks(y_scale)
ax.grid()
ax.legend()

# Bottom subplot - Tuned model
ax = axes[1]
ax.set_title("Percent Error for Tuned Model")
ax.bar(rmse_data_no_avg.index - 0.1, rmse_data_no_avg["Tuned_TrainData_RMSE_Percent_Error"], width=0.2, label="Tuned Model Training Data Percent Error", color="green")
ax.bar(rmse_data_no_avg.index + 0.1, rmse_data_no_avg["Tuned_TestData_RMSE_Percent_Error"], width=0.2, label="Tuned Model Testing Data Percent Error", color="red")
ax.set_xlabel("Fold Index")
ax.set_ylabel("Percent Error (%)")
ax.set_yticks(y_scale)
ax.grid()
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images/MLP_Percent_Error_Comparison.png")

# TODO: Add plots for average difference in predicted vs actual tensile_strength values

# Calculating average differences
default = []
tuned = []

for i in range(1, 11):
    train_col = f"fold_{i}_train"
    test_col = f"fold_{i}_test"
    default_train_mean = default_MLP_train_diff[train_col].mean(skipna=True)
    default_test_mean = default_MLP_train_diff[test_col].mean(skipna=True)
    tuned_train_mean = tuned_MLP_train_diff[train_col].mean(skipna=True)
    tuned_test_mean = tuned_MLP_train_diff[test_col].mean(skipna=True)

    default.append({
            "fold": i,
            "train_mean": default_train_mean,
            "test_mean": default_test_mean,
        })
    tuned.append({
            "fold": i,
            "train_mean": tuned_train_mean,
            "test_mean": tuned_test_mean,
        })

print("Default Model Average Differences:", default)
print("Tuned Model Average Differences:", tuned)

plt.figure(figsize=(10, 6))
plt.plot([d["fold"] for d in default], [d["train_mean"] for d in default], marker="o", label="Default Model Training Mean Difference", color="blue")
plt.plot([d["fold"] for d in default], [d["test_mean"] for d in default], marker="o", label="Default Model Testing Mean Difference", color="orange")
plt.plot([d["fold"] for d in tuned], [d["train_mean"] for d in tuned], marker="o", label="Tuned Model Training Mean Difference", color="green")
plt.plot([d["fold"] for d in tuned], [d["test_mean"] for d in tuned], marker="o", label="Tuned Model Testing Mean Difference", color="red")
plt.title("MLP Regressor Average Difference in Predicted vs Actual Tensile Strength")
plt.xlabel("Fold Index")
plt.ylabel("Average Difference in Tensile Strength")
plt.legend()
plt.grid()
plt.savefig("images/MLP_Average_Difference_Comparison.png")


# TODO: Add hyperparameter impact analysis plots
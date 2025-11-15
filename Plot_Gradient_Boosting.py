import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rmse_data = pd.read_csv("datasets/Gradient_Boosting_RMSE_PercentErrors.csv")
rmse_data_no_avg = rmse_data[:-1] # Exclude the last row which contains averages

# Plotting RMSE for Default and Tuned models
plt.figure(figsize=(10, 6))
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Default_TestData_RMSE"], marker="o", label="Default Model Test RMSE", color="blue")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Tuned_TestData_RMSE"], marker="o", label="Tuned Model Test RMSE", color="orange")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Default_TrainData_RMSE"], marker="o", label="Default Model Train RMSE", color="green")
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg["Tuned_TrainData_RMSE"], marker="o", label="Tuned Model Train RMSE", color="red")
plt.title("Gradient Boosting Regressor Root Mean Squared Error Comparison")
plt.xlabel("Fold Index")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend()
plt.grid()
plt.savefig("images/Gradient_Boosting_RMSE_Comparison.png")
plt.show()


y_scale = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5]
# Plotting Percent Error for Default and Tuned models
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("Gradient Boosting Regressor Percent Error Comparison")
fig.subplots_adjust(hspace=0.4)  # add vertical space between subplots

# Top subplot - Default model
ax = axes[0]
ax.set_title("Percent Error for Default Model")
ax.bar(rmse_data_no_avg.index - 0.1, rmse_data_no_avg["Default_TrainData_RMSE_Percent_Error"], width=0.2, label="Default Model Train Data Percent Error", color="blue")
ax.bar(rmse_data_no_avg.index + 0.1, rmse_data_no_avg["Default_TestData_RMSE_Percent_Error"], width=0.2, label="Default Model Test Data Percent Error", color="orange")
ax.set_xlabel("Fold Index")
ax.set_ylabel("Percent Error (%)")
ax.set_yticks(y_scale)
ax.grid()
ax.legend()

# Bottom subplot - Tuned model
ax = axes[1]
ax.set_title("Percent Error for Tuned Model")
ax.bar(rmse_data_no_avg.index - 0.1, rmse_data_no_avg["Tuned_TrainData_RMSE_Percent_Error"], width=0.2, label="Tuned Model Train Data Percent Error", color="green")
ax.bar(rmse_data_no_avg.index + 0.1, rmse_data_no_avg["Tuned_TestData_RMSE_Percent_Error"], width=0.2, label="Tuned Model Test Data Percent Error", color="red")
ax.set_xlabel("Fold Index")
ax.set_ylabel("Percent Error (%)")
ax.set_yticks(y_scale)
ax.grid()
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images/Gradient_Boosting_Percent_Error_Comparison.png")
plt.show()

# TODO: Add plots for average difference in predicted vs actual tensile_strength values
# TODO: Add hyperparameter impact analysis plots

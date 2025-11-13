import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rmse_data = pd.read_csv('datasets/Gradient_Boosting_RMSE_PercentErrors.csv')
rmse_data_no_avg = rmse_data[:-1] # Exclude the last row which contains averages

# Plotting RMSE for Default and Tuned models
plt.figure(figsize=(10, 6))
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg['Default_TestData_RMSE'], marker='o', label='Default Model Test RMSE', color='blue')
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg['Tuned_TestData_RMSE'], marker='o', label='Tuned Model Test RMSE', color='orange')
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg['Default_TrainData_RMSE'], marker='o', label='Default Model Train RMSE', color='green')
plt.plot(rmse_data_no_avg.index, rmse_data_no_avg['Tuned_TrainData_RMSE'], marker='o', label='Tuned Model Train RMSE', color='red')
plt.title('Gradient Boosting Regressor Root Mean Squared Error Comparison')
plt.xlabel('Fold Index')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend()
plt.grid()
plt.show()
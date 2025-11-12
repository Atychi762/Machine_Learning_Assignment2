import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('datasets/steel.csv')

scaler = StandardScaler()
feature_cols = dataset.select_dtypes(include=[np.number]).columns.drop('tensile_strength')
dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

kf = KFold(n_splits=10, shuffle=False)
average_mean_squared_error = 0

for train, test in kf.split(dataset):
    train_data = dataset.iloc[train]
    test_data = dataset.iloc[test]

    X_train = train_data.drop('tensile_strength', axis=1)
    y_train = train_data['tensile_strength']
    X_test = test_data.drop('tensile_strength', axis=1)
    y_test = test_data['tensile_strength']

    model = gbr()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    average_mean_squared_error += mse

average_mean_squared_error /= 10
print(f'\nAverage Mean Squared Error over 10 folds: {average_mean_squared_error}')
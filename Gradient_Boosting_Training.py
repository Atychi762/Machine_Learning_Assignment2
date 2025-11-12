import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import KFold

dataset = pd.read_csv('datasets/steel.csv')
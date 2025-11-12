import pandas as pd
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.model_selection import KFold

dataset = pd.read_csv('datasets/steel.csv')
import numpy as np
import pandas as pd

X_test = pd.read_csv('../../data/X_test_update.csv', index_col=0)
X_train = pd.read_csv('../../data/X_train_update.csv', index_col=0)
Y_train = pd.read_csv('../../data/Y_train_CVw08PX.csv', index_col=0)
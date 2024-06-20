import pandas as pd
from exploration import exploration_donnee
from visualisation import data_visualisation

# Importer les fichiers CSV
X_train = pd.read_csv('data/X_train.csv', index_col=0)
Y_train = pd.read_csv('data/Y_train.csv', index_col=0)
X_test = pd.read_csv('data/X_test.csv', index_col=0)

# Fusionner X_train (variables explicatives) et y_train (variable cible)
fusion = pd.merge(X_train, Y_train, left_index=True, right_index=True)

# Exploration des données
exploration_donnee(fusion)

# Data visualisation
data_visualisation(fusion)
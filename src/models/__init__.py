import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Importer les fichiers CSV
X = pd.read_csv('data/X_train.csv', index_col=0)
y = pd.read_csv('data/Y_train.csv', index_col=0)

#Répartition en données d'entrainement et en données de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

#print(X_train.head())


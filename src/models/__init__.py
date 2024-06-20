import pandas as pd
from preprocessing import *

# Pre-processing
print("Etape Fusion:\n")
X, y = fusion_description_designation()
X_train, y_train, X_test, y_test = re_echantillonage(X,y)
X['descriptif'] = X.apply(lambda row: clean_column_descriptif(row['descriptif']), axis=1)

# Créer un fichier CSV à partir de X
X.to_csv('data/X_preprocessed.csv')

print("Etape analyse langues:\n")
df_lang = traitement_lang(X)

print("Etape Echantillonnage:\n")
X_train, X_test, y_train, y_test = re_echantillonage(X,y)
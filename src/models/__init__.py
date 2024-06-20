import pandas as pd
from preprocessing import *

# Pre-processing
X, y = fusion_description_designation()
re_echantillonage(X,y)
X['descriptif'] = X.apply(lambda row: clean_column_descriptif(row['descriptif']), axis=1)

# Créer un fichier CSV à partir de X
X.to_csv('data/X_preprocessed.csv')

# Vérifier la présence de doublon dans les images
# image_folder1 = r'../../data/images/image_test'
# image_folder2 = r'../../data/images/image_train'

# df1 = pd.DataFrame({'image_exists': []})
# df1['image_exists'] = df1.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder1), axis=1)

# df2 = pd.DataFrame({'image_exists': []})
# df2['image_exists'] = df2.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder2), axis=1)

# print(df1['image_exists'].value_counts())
# print(df2['image_exists'].value_counts())
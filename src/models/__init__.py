import pandas as pd
from preprocessing import fusion_description_designation, re_echantillonage

# Pre-processing
X, y = fusion_description_designation()
re_echantillonage(X,y)

# Vérifier la présence de doublon dans les images
# image_folder1 = r'../../data/images/image_test'
# image_folder2 = r'../../data/images/image_train'

# df1 = pd.DataFrame({'image_exists': []})
# df1['image_exists'] = df1.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder1), axis=1)

# df2 = pd.DataFrame({'image_exists': []})
# df2['image_exists'] = df2.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder2), axis=1)

# print(df1['image_exists'].value_counts())
# print(df2['image_exists'].value_counts())
import pandas as pd
from preprocessing import fusion_description_designation, re_echantillonage, traitement_lang

# Pre-processing
print("Etape Fusion:\n")
X, y = fusion_description_designation()

print("Etape analyse langues:\n")
df_lang = traitement_lang(X)

print("Etape Echantillonnage:\n")
X_train, X_test, y_train, y_test = re_echantillonage(X,y)

#test :: échantillonnage X2 pour réduire l'échantillon de X_test
#X_train, X_test, y_train, y_test = re_echantillonage(X_test,y_test)
#traitement_lang(X_test)


# Vérifier la présence de doublon dans les images
# image_folder1 = r'../../data/images/image_test'
# image_folder2 = r'../../data/images/image_train'

# df1 = pd.DataFrame({'image_exists': []})
# df1['image_exists'] = df1.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder1), axis=1)

# df2 = pd.DataFrame({'image_exists': []})
# df2['image_exists'] = df2.apply(lambda row: check_image_exists(row['imageid'],row['productid'], image_folder2), axis=1)

# print(df1['image_exists'].value_counts())
# print(df2['image_exists'].value_counts())
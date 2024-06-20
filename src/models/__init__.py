import pandas as pd
from preprocessing import fusion_description_designation, re_echantillonage
from collections import Counter
import re
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
yy = pd.DataFrame()
yy['prdtypecode']=y['prdtypecode']
yy['descriptif'] = X['descriptif']
print(yy)
def nettoyer_et_separer(description):
    description = description.lower()  # convertir en minuscules
    mots = re.findall(r'\b\w+\b', description)  # extraire les mots
    return mots

# Appliquer la fonction de nettoyage à chaque ligne de la colonne 'designation'
yy['mots'] = yy['descriptif'].apply(nettoyer_et_separer)

# Initialiser un dictionnaire vide pour stocker les compteurs par classe
compteurs_par_classe = {}

# Parcourir chaque groupe de classe
for classe, groupe in yy.groupby('prdtypecode'):
    # Combiner tous les mots de la classe en une seule liste
    tous_les_mots = [mot for liste_mots in groupe['mots'] for mot in liste_mots]
    # Compter les occurrences de chaque mot
    compteurs_par_classe[classe] = Counter(tous_les_mots)

# Créer un DataFrame à partir du dictionnaire de compteurs
df_resultat = pd.DataFrame(compteurs_par_classe).fillna(0).astype(int)
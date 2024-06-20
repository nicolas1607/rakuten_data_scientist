import re
import pandas as pd
from preprocessing import *
from collections import Counter

# Pre-processing
print("Fusion des colonnes description et designation :\n")
X, y = fusion_description_designation()

print("Analyse des langues dans la colonne descriptif :\n")
df_lang = traitement_lang(X)

print("Ré-échantillonnage du jeu de donnée :\n")
X_train, X_test, y_train, y_test = re_echantillonage(X,y)

print("Nettoyage de la colonne descriptif:\n")
X['descriptif'] = X.apply(lambda row: clean_column_descriptif(row['descriptif']), axis=1)
X.to_csv('data/X_preprocessed.csv')

print("Création du DataFrame (nombre d'occurence des mots en fonction du prdtypecode) :\n")
yy = pd.DataFrame()
yy['prdtypecode']=y['prdtypecode']
yy['descriptif'] = X['descriptif']

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
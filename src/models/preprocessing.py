'''
Changement à prendre en compte :
    
    Riadh : Identifier les mots les plus fréquents pour chacune des classes avec un histogramme, un nuage de point ou un nuage de mots

    Slimane : Étudier les langues présentes dans vos descriptions et désignations. Cette tâche vous aidera à définir quelle langue ou quelles langues utiliser dans la partie prétraitement
    
    Nicolas : Comme vous pouvez le voir, le texte est écrit par un humain, et il vous faudra utiliser du NLP pour rendre votre texte compréhensible pour votre modèle. Pour cela, les étapes possibles à ce stade sont : la tokenisation, le retrait des espaces excessifs, la mise en minuscules, la suppression des balises HTML, le retrait des nombres et des caractères spéciaux, supprimer les stopwords, les mots de moins de 3 lettres et les balises HTML, le filtrage des mots vides, la racinisation et la lemmatisation, la vectorisation...
    
    Simplice : Vous avez à disposition plusieurs classes "ambiguës" et ne savez pas à quoi correspond chacune d'elles. Vous pouvez essayer d'identifier le groupe de mots désignant chaque classe afin d'avoir une idée plus claire sur le contenu de chacune des classes.
    
    Autre : Étudier la répartition de la fréquence des mots dans votre colonne texte
'''

import os
import re
import pandas as pd
from language_labels import language_labels
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from langdetect import detect, lang_detect_exception, DetectorFactory
from collections import Counter
from nltk.stem.snowball import FrenchStemmer
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0

def check_image_exists(imageid, productid, folder):
    image_filename = f'image_{imageid}_product_{productid}.jpg'
    image_path = os.path.join(folder, image_filename)
    return os.path.isfile(image_path)

def fusion_description_designation():
    X = pd.read_csv('data/X_train.csv', index_col=0)
    y = pd.read_csv('data/Y_train.csv', index_col=0)
    X['descriptif'] = X['description'].astype(str).replace("nan", "") + " " + X['designation'].astype(str)
    X = X.drop(['designation', 'description'], axis=1)
    return X, y

def re_echantillonage(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    return X_train, X_test, y_train, y_test

def detect_lang(texte):
    try:
        langue = detect(texte) #code de la langue
        return language_labels.get(langue, "Langue non supportée")
    except lang_detect_exception.LangDetectException:
        return "Erreur_langue"

def traitement_lang(X):
    df_lang=pd.DataFrame(X['descriptif'].apply(detect_lang))
    print(df_lang.value_counts())
    print("nombre d'éléments=", len(df_lang))
    return df_lang

def clean_column_descriptif(texte):

    words = set(stopwords.words('french'))

    # Tokenisation
    # texte = word_tokenize(texte)

    # Retrait des espaces excessifs
    texte = re.sub("\s{2,}", " ", texte)

    # Mettre en minuscule
    texte = texte.lower()

    # Supprimer les noms de fichiers (jpg, jpeg, png, gif, pdf)
    # texte = re.sub(r'\b\w+\.(jpg|jpeg|png|gif|pdf)\b', '', texte)

    # Supprimer les URLs
    # texte = re.sub(r'http\S+', '', texte)

    # Supprimer les balises HTML
    texte = BeautifulSoup(texte, "html.parser").get_text()

    # Supprimer les nombres et caractères spéciaux
    texte = re.sub('\d+', '', texte)
    texte = re.sub("[^a-zA-Z]", " ", texte)

    # On traduit chaques mots de texte en français
    # print(texte)
    # texte = GoogleTranslator(source='auto', target='fr').translate(texte)

    # Supprimer les stopwords et les mots de moins de 3 lettres
    texte = ' '.join([word for word in texte.split() if word not in words])
    texte = ' '.join([word for word in texte.split() if len(word) > 2])

    # print(texte)
    # print()

    # Racinisation et lemmatisation
    # stemmer = FrenchStemmer()
    # texte = ' '.join([stemmer.stem(word) for word in texte.split()])

    return texte

def nettoyer_et_separer(description):
    mots = re.findall(r'\b\w+\b', description)  # extraire les mots
    return mots

def word_occurence_by_prdtypecode(X, y):
    df = pd.DataFrame()
    df['prdtypecode']= y['prdtypecode']
    df['descriptif'] = X['descriptif']

    # Appliquer la fonction de nettoyage à chaque ligne de la colonne 'designation'
    df['mots'] = df['descriptif'].apply(nettoyer_et_separer)

    # Initialiser un dictionnaire vide pour stocker les compteurs par classe
    compteurs_par_classe = {}

    # Parcourir chaque groupe de classe
    for classe, groupe in df.groupby('prdtypecode'):
        # Combiner tous les mots de la classe en une seule liste
        tous_les_mots = [mot for liste_mots in groupe['mots'] for mot in liste_mots]
        # Compter les occurrences de chaque mot
        compteurs_par_classe[classe] = Counter(tous_les_mots)

    # Créer un DataFrame à partir du dictionnaire de compteurs
    df_result = pd.DataFrame(compteurs_par_classe).fillna(0).astype(int)
    return df_result

def pre_processing():
    print("Fusion des colonnes description et designation\n")
    X, y = fusion_description_designation()

    print("Analyse des langues dans la colonne descriptif\n")
    # df_lang = traitement_lang(X)

    print("Nettoyage de la colonne descriptif\n")
    if os.path.exists("data/X_preprocessed.csv"):
        X = pd.read_csv('data/X_preprocessed.csv')
        X = X.drop('Unnamed: 0', axis=1)
        X = X.fillna('')
    else:
        X['descriptif'] = X.apply(lambda row: clean_column_descriptif(row['descriptif']), axis=1)
        X.to_csv('data/X_preprocessed.csv')

    print("Création du DataFrame (nombre d'occurence des mots en fonction du prdtypecode)\n")
    df_result = word_occurence_by_prdtypecode(X, y)

    print("Ré-échantillonnage du jeu de donnée\n")
    X_train, X_test, y_train, y_test = re_echantillonage(X,y)

    return X_train, X_test, y_train, y_test
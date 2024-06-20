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
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.model_selection import train_test_split
from langdetect import detect, lang_detect_exception, DetectorFactory

DetectorFactory.seed = 0

def fusion_description_designation():
    X = pd.read_csv('data/X_train.csv', index_col=0)
    y = pd.read_csv('data/Y_train.csv', index_col=0)
    X['descriptif'] = X['description'].astype(str).replace("nan", "") + " " + X['designation'].astype(str)
    X = X.drop(['designation', 'description'], axis=1)
    return X, y

def re_echantillonage(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    return X_train, X_test, y_train, y_test

def check_image_exists(imageid, productid, folder):
    image_filename = f'image_{imageid}_product_{productid}.jpg'
    image_path = os.path.join(folder, image_filename)
    return os.path.isfile(image_path)

def detect_lang(texte):
    try:
        langue = detect(texte) #code de la langue
        #return langue
        return language_labels.get(langue, "Langue non supportée")
    except lang_detect_exception.LangDetectException:
        return "Erreur_langue"

def traitement_lang(X):
    df_lang=pd.DataFrame(X['descriptif'].apply(detect_lang))
    print(df_lang.value_counts())
    print("nombre d'éléments=", len(df_lang))
    return df_lang

def clean_column_descriptif(row):

    words = set(stopwords.words('french'))
            
    texte = row

    # Tokenisation
    # texte = word_tokenize(texte)

    # Retrait des espaces excessifs
    texte = re.sub("\s{2,}", " ", row)

    # Mettre en minuscule
    texte = texte.lower()

    # Supprimer les noms de fichiers (jpg, jpeg, png, gif, pdf)
    # texte = re.sub(r'\b\w+\.(jpg|jpeg|png|gif|pdf)\b', '', texte)

    # Supprimer les URLs
    # texte = re.sub(r'http\S+', '', texte)

    # Supprimer les balises HTML
    texte = BeautifulSoup(texte, "html.parser").get_text()

    # Supprimer les nombres et caractères spéciaux
    texte = re.sub(r'\d+', '', texte)
    texte = re.sub("[^a-zA-Z]", " ", texte)

    # Supprimer les stopwords et les mots de moins de 3 lettres
    texte = ' '.join([word for word in texte.split() if word not in words])
    texte = ' '.join([word for word in texte.split() if len(word) > 2])

    # Racinisation et lemmatisation
    # stemmer = FrenchStemmer()
    # texte = ' '.join([stemmer.stem(word) for word in texte.split()])

    return texte

language_labels = {'af':'Afrikaans',
'ar':'Arabe',
'bg':'Bulgare',
'bn':'Bengali',
'ca':'Catalan',
'cs':'Tchèque',
'cy':'Gallois',
'da':'Danois',
'de':'Allemand',
'el':'Grec',
'en':'Anglais',
'es':'Espagnol',
'et':'Estonien',
'fa':'Persan',
'fi':'Finnois',
'fr':'Français',
'gu':'Gujarati',
'he':'Hébreu',
'hi':'Hindi',
'hr':'Croate',
'hu':'Hongrois',
'id':'Indonésien',
'it':'Italien',
'ja':'Japonais',
'kn':'Kannada',
'ko':'Coréen',
'lt':'Lituanien',
'lv':'Letton',
'mk':'Macédonien',
'ml':'Malayalam',
'mr':'Marathi',
'ne':'Népalais',
'nl':'Néerlandais',
'no':'Norvégien',
'pa':'Pendjabi',
'pl':'Polonais',
'pt':'Portugais',
'ro':'Roumain',
'ru':'Russe',
'sk':'Slovaque',
'sl':'Slovène',
'so':'Somali',
'sq':'Albanais',
'sv':'Suédois',
'sw':'Swahili',
'ta':'Tamoul',
'te':'Télougou',
'th':'Thaï',
'tl':'Tagalog',
'tr':'Turc',
'uk':'Ukrainien',
'ur':'Ourdou',
'vi':'Vietnamien',
'zh-cn':'Chinois simplifié',
'zh-tw':'Chinois tradition' 
}
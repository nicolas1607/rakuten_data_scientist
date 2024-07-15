import os
import re
import cv2
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from language_labels import language_labels
from langdetect import detect, lang_detect_exception, DetectorFactory
from bs4 import BeautifulSoup
from collections import Counter
from unidecode import unidecode
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
from tqdm import tqdm
from scipy import sparse
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

tqdm.pandas()

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

    df = pd.merge(X, y, left_index=True, right_index=True)

    return df

def detect_lang(texte):
    try:
        langue = detect(texte) 
        return language_labels.get(langue, "Langue non supportée")
    except lang_detect_exception.LangDetectException:
        return "Erreur_langue"

def clean_column_descriptif(texte):

    # Retrait des espaces excessifs
    texte = re.sub("\s{2,}", " ", texte)

    # Mettre en minuscule
    texte = texte.lower()

    # Supprimer les balises HTML
    texte = BeautifulSoup(texte, "html.parser").get_text()

    # Supprimer les nombres 
    texte = re.sub('\d+', '', texte)

    # Supprimer les accents
    texte = unidecode(texte)

    # Supprimer les caractères spéciaux
    texte = re.sub("[^a-zA-Z]", " ", texte)

    # Supprimer les stopwords et les mots de moins de 4 lettres
    stop_words = set(stopwords.words())
    texte = ' '.join([word for word in texte.split() if word not in stop_words])
    texte = ' '.join([word for word in texte.split() if len(word) > 3])

    return texte

def word_occurence_by_prdtypecode(df):

    # Appliquer la fonction de nettoyage à chaque ligne de la colonne 'designation'
    df['mots'] = df['descriptif_cleaned'].apply(lambda description: re.findall(r'\b\w+\b', description))

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

def nuage_de_mots(df_result):
    for classe in df_result.columns:
        wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(df_result[classe])
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Nuage de mots pour la classe {classe}')
        plt.savefig(f"reports/figures/nuage_de_mot/{classe}.png", bbox_inches='tight')

def translate(texte):
    if len(texte) > 10000:
        print("Texte trop long pour la traduction :", len(texte))
        texte1 = GoogleTranslator(source=detect(texte), target='fr').translate(texte[:4000])
        texte2 = GoogleTranslator(source=detect(texte), target='fr').translate(texte[4000:8000])
        texte3 = GoogleTranslator(source=detect(texte), target='fr').translate(texte[8000:12000])
        texte = texte1 + texte2 + texte3
    elif len(texte) > 5000:
        print("Texte trop long pour la traduction :", len(texte))
        texte1 = GoogleTranslator(source=detect(texte), target='fr').translate(texte[:4000])
        texte2 = GoogleTranslator(source=detect(texte), target='fr').translate(texte[4000:8000])
        texte = texte1 + texte2
    else:
        texte = GoogleTranslator(source=detect(texte), target='fr').translate(texte)
    return texte

def resample_data(X_train, y_train, booOverSampling):   
    
    print("Dimensions avant ré-échantillonnage:")
    print("X_train=", X_train.shape)
    print("y_train=", y_train.shape)

    if (not booOverSampling):                
        LibMethode = "sous-échantillonnage"
        r_ech = EditedNearestNeighbours()
    else:
        LibMethode = "sur-échantillonnage"
        r_ech = ADASYN()
    
    print("\nMéthode de", LibMethode,":", r_ech)
    
    X_train, y_train = r_ech.fit_resample(X_train, y_train)
    sparse.save_npz("data/X_train_sampled.npz", X_train)
    y_train.to_csv('data/y_train_sampled.csv', index=False)

    print("\Dimensions après ré-échantillonnage")
    print("X_train_r=", X_train.shape)
    print("y_train_r=", y_train.shape)
    print()

    return X_train, y_train

def pre_processing_texte(tokenizer_name=None, isResampling=False):

    print("Fusion des colonnes description et designation\n")
    df = fusion_description_designation()

    print("Analyse des langues dans la colonne descriptif\n")
    if os.path.exists("data/df_lang_preprocessed.csv"):
        df_lang = pd.read_csv('data/df_lang_preprocessed.csv')
    else:
        df_lang = pd.DataFrame(df['descriptif'].progress_apply(detect_lang))
        df_lang.to_csv('data/df_lang_preprocessed.csv')
        print(df_lang.value_counts())
        print("nombre d'éléments total =", len(df_lang))
    
    print("Nettoyage de la colonne descriptif\n")
    if os.path.exists("data/df_cleaned.csv"):
        df = pd.read_csv('data/df_cleaned.csv')
        df = df.fillna('')
    else:
        df['descriptif_cleaned'] = df['descriptif'].progress_apply(clean_column_descriptif)
        df.to_csv('data/df_cleaned.csv')

    # print("Traduction de la colonne descriptif\n")
    # if os.path.exists("data/df_traducted.csv"):
    #     df = pd.read_csv('data/df_traducted.csv')
    #     df = df.fillna('')
    # else:
    #     df['descriptif_trad'] = df['descriptif_cleaned'].progress_apply(lambda row: translate(row), axis=1)
    #     df.to_csv('data/df_traducted.csv')

    print("Lemmatisation de la colonne descriptif\n")
    if os.path.exists("data/df_lemmatized.csv"):
        df = pd.read_csv('data/df_lemmatized.csv')
        df = df.fillna('')
    else:
        lemmatizer = WordNetLemmatizer()
        df['descriptif_cleaned'] = df['descriptif_cleaned'].progress_apply(lambda texte: ' '.join([lemmatizer.lemmatize(mot) for mot in texte.split()]))
        df.to_csv('data/df_lemmatized.csv')

    print("Tokenisation de la colonne descriptif\n")
    if tokenizer_name != 'bert':
        if os.path.exists("data/df_tokenized.csv"):
            df = pd.read_csv('data/df_tokenized.csv')
            df = df.fillna('')
        else:
            df['tokens'] = df['descriptif_cleaned'].progress_apply(word_tokenize)
            df.to_csv('data/df_tokenized.csv')
            df = pd.read_csv('data/df_tokenized.csv')
            df = df.fillna('')

    print("Nuage de mots pour les 27 catégories\n")
    if len(os.listdir('reports/figures/nuage_de_mot')) == 0:
        df_result = word_occurence_by_prdtypecode(df)
        nuage_de_mots(df_result)

    # df = df.sample(frac=0.005, random_state=66)

    print("Répartition Train/Test du jeu de donnée\n")
    if tokenizer_name != 'bert':
        X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['prdtypecode'], test_size=0.2, random_state=66)
    else:
        X_train, X_test, y_train, y_test = train_test_split(df['descriptif_cleaned'], df['prdtypecode'], test_size=0.2, random_state=66)

    print("Vectorisation de la colonne descriptif\n")
    if tokenizer_name != 'bert':
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    
    print("Ré-échantillonnage du jeu de donnée\n")
    if isResampling == True:
        if os.path.exists('data/X_train_sampled.npz') and os.path.exists('data/y_train_sampled.csv') :
            X_train = sparse.load_npz("data/X_train_sampled.npz")
            y_train = pd.read_csv('data/y_train_sampled.csv')
        else:
            X_train, y_train = resample_data(X_train, y_train, booOverSampling=True)

    return X_train, X_test, y_train, y_test

def pre_processing_image(size):

    input_path = 'data/images/image_train/'
    output_path = 'data/images/image_train_preprocessed/'

    resize_images_folder(input_path, output_path, size)

    X = pd.read_csv('data/X_train.csv', index_col=0)
    y = pd.read_csv('data/Y_train.csv', index_col=0)

    df = X.merge(y, left_index=True, right_index=True)
    df['filepath'] = df.apply(lambda row: output_path + 'image_' + str(row['imageid']) + '_product_' + str(row['productid']) + '.jpg', axis=1)
    df['prdtypecode'] = df['prdtypecode'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(df['filepath'], df['prdtypecode'], test_size=0.20, random_state=66)

    return X_train, X_test, y_train, y_test

def resize_images_folder(input_path, output_path, size):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if len(os.listdir(output_path)) == 0:
        for index, filename in enumerate(os.listdir(input_path)):
            image = cv2.imread(input_path+filename, cv2.IMREAD_COLOR)

            # Passage en noir et blanc
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Tester le filtre : flou gaussien

            # Erosion de l'image
            # filtre = cv2.GaussianBlur(image, ksize = (3,3), sigmaX = 0)
            # kernel = np.ones((3,3), np.uint8)
            # image = cv2.erode(filtre, kernel)

            if image is not None:
                image = cv2.resize(image, (size, size))
                cv2.imwrite(output_path+filename, image)
            else:
                print(f"Erreur de lecture de l'image : {input_path+filename}")

            if index % 1000 == 0:
                print(f"Redimensionnement de {index} images terminé")
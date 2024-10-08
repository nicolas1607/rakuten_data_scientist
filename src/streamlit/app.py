import os
import sys
import pickle
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score
from keras.preprocessing.image import ImageDataGenerator
from src.models.model_res_net_50 import data_augmentation
from src.models.preprocessing import pre_processing_texte, pre_processing_image

@st.cache_data

def load_data():
    if not os.path.exists('data/texte_preprocessed'):
        X_train_texte, X_test_texte, y_train_texte, y_test_texte, _, _ = pre_processing_texte(isResampling=False)
    else:
        X_train_texte = sparse.load_npz("data/texte_preprocessed/X_train.npz")
        X_test_texte = sparse.load_npz("data/texte_preprocessed/X_test.npz")
        y_train_texte = pd.read_csv('data/texte_preprocessed/y_train.csv')
        y_test_texte = pd.read_csv('data/texte_preprocessed/y_test.csv')

    if not os.path.exists('data/image_preprocessed'):
        X_train_image, X_test_image, y_train_image, y_test_image = pre_processing_image(size=125)
    else:
        X_train_image = pd.read_csv('data/image_preprocessed/X_train.csv')
        X_test_image = pd.read_csv('data/image_preprocessed/X_test.csv')
        y_train_image = pd.read_csv('data/image_preprocessed/y_train.csv')
        y_test_image = pd.read_csv('data/image_preprocessed/y_test.csv')
    return X_train_texte, X_test_texte, y_train_texte, y_test_texte, X_train_image, X_test_image, y_train_image, y_test_image

def prediction(classifier):
    model_mapping = {
        'LogisticRegression': 'logistic_regression',
        'MultinomialNB': 'multinomialNB',
        'ComplementNB': 'complementNB',
        'LinearSVM': 'linear_svm',
        'SGDClassifier': 'sgd',
        'DecisionTreeClassifier': 'decisionTree',
        'Sequential (faible augmentation des données)': 'sequential',
        'ResNet50 (forte augmentation des données)': 'resnet50'
    }
    model_name = model_mapping[classifier]
    model = pickle.load(open(f'models/{model_name}.pkl', 'rb'))
    return model, model_name

def scores_texte(model, choice, X_test_texte, y_test_texte):
    if choice == 'Scores de performance':
        return round(model.score(X_test_texte, y_test_texte), 4), round(f1_score(y_test_texte, model.predict(X_test_texte), average='weighted'), 4)
    elif choice == 'Matrice de confusion':
        return confusion_matrix(y_test_texte, model.predict(X_test_texte))

def scores_image(model_name):

    if os.path.exists(f"reports/figures/{model_name}/results.pkl"):
        results = pickle.load(open(f'reports/figures/{model_name}/results.pkl', 'rb'))
    else:
        X_train_image_list = X_train_image['filepath'].tolist() if isinstance(X_train_image, pd.DataFrame) else X_train_image
        y_train_image_list = y_train_image['prdtypecode'].astype(str).tolist() if isinstance(y_train_image, pd.DataFrame) else y_train_image
        X_test_image_list = X_test_image['filepath'].tolist() if isinstance(X_test_image, pd.DataFrame) else X_test_image
        y_test_image_list = y_test_image['prdtypecode'].astype(str).tolist() if isinstance(y_test_image, pd.DataFrame) else y_test_image

        train_df = pd.DataFrame({'filepath': X_train_image_list, 'prdtypecode': y_train_image_list})
        test_df = pd.DataFrame({'filepath': X_test_image_list, 'prdtypecode': y_test_image_list})
        
        _, test_generator = data_augmentation(model_name, train_df, test_df, 8, 125)

        if model_name == 'sequential':
            model = pickle.load(open("models/sequential.pkl", "rb"))
        else:
            model = pickle.load(open("models/resnet50.pkl", "rb"))
        
        results = model.evaluate(test_generator)
        results = dict(zip(model.metrics_names, results))
        pickle.dump(results, open(f'reports/figures/{model_name}/results.pkl', 'wb'))

    return results['loss'], results['accuracy'], results['f1_score']

def get_dataframe_texte(X_train, Y_train):
    fusion = pd.merge(X_train, Y_train, left_index=True, right_index=True)
    return fusion

def get_dataframe_image(X, y, output_path):
    df = X.merge(y, left_index=True, right_index=True)
    df['filepath'] = df.apply(lambda row: output_path + 'image_' + str(row['imageid']) + '_product_' + str(row['productid']) + '.jpg', axis=1)
    df['prdtypecode'] = df['prdtypecode'].astype(str)
    df = df[['filepath', 'prdtypecode']]
    return df

def data_augmentation(model_name, train_df, test_df, batch_size, size):

    if model_name == 'sequential':
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
        test_datagen = ImageDataGenerator(rescale=1./255)
    elif model_name == 'resnet50':
        train_datagen = ImageDataGenerator(
            rescale=0.1,
            zoom_range=0.1,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )

        test_datagen = ImageDataGenerator(rescale=0.1)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

X_train = pd.read_csv('data/X_train.csv', index_col=0)
Y_train = pd.read_csv('data/Y_train.csv', index_col=0)
fusion = get_dataframe_texte(X_train, Y_train)

X_train_texte, X_test_texte, y_train_texte, y_test_texte, X_train_image, X_test_image, y_train_image, y_test_image = load_data()

### Organisation du Streamlit ###

st.title("Challenge Rakuten")
st.sidebar.title("Challenge Rakuten")

pages = [
    "Introduction", # simplice
    "Exploration des données", # riadh
    "Visualisation des données", # slimane
    "Pre-processing", #nicolas
    "Classification des textes", # slimane
    "Classification des images", # nicolas
    "Interprétation des résultats", # simplisse
    "Conclusion" # riadh
]

page = st.sidebar.radio("", pages)
st.sidebar.warning("Cohorte : Bootcamp DS de mai 2024\n\n**Nicolas Mormiche**\nhttps://linkedin.com/in/mormichen\n\n**Riadh Zidi**\nhttps://www.linkedin.com/in/riadh-zidi-493800276/\n\n**Simplice Lolo Mvoumbi**\nhttps://www.linkedin.com/in/simplice-lolo-mvoumbi-726606286/\n\n**Slimane Chelouah**\nhttp://www.linkedin.com/in/slimane-chelouah")

### Introduction ###

if page == pages[0]:

    st.markdown("## Introduction")
    st.image("reports/figures/challenge_data.png")
    st.write("**Challenge Rakuten** : https://challengedata.ens.fr/participants/challenges/35/")
    st.write("")
    st.write("Ce projet s'inscrit dans le cadre de nos travaux de fin de formation, portant sur la classification multimodale de produits e-commerce (texte et image). Il consiste à prédire le code type des produits du catalogue de Rakuten France.")
    st.write("")
    st.write("Le projet utilise les données fournies par Rakuten :")
    st.write("- 84916 descriptions de produits")
    st.write("- 84916 images")
    st.write("- 27 catégories de produits uniques")
    st.write("")
    st.write("L'objectif est d'étudier la classification multimodale et d'améliorer les f1-score obtenus par Rakuten qui utilise un CNN simplifié pour les texte et un Residual Networks (ResNet50) pour les images :")
    st.write("- Classification des textes avec CNN : 0.8113")
    st.write("- Classification des images avec ResNet50 : 0,5534")
    st.write("")
    st.write("Voici un exemple de représentation de produit e-commerce Rakuten :")
    st.write("- Catégorie de produit : " + str(fusion['prdtypecode'][7]))
    st.write("- Description : " + fusion['designation'][7])
    st.write("- Image :")
    st.image("data/images/image_train/image_"+str(fusion['imageid'][7])+"_product_"+str(fusion['productid'][7])+".jpg")

### Exploration des données ###    

if page == pages[1]:

    st.write("## Exploration des données")

    st.write("#### 1. Aperçu des données")
    st.write("")
    st.write("Voici un aperçu des premières lignes du jeu de données :")
    st.dataframe(fusion.head())
    st.write("")
    st.write("On remarque donc 5 colonnes avec :")
    st.write("- designation : titre du produit, court texte résumant le produit")
    st.write("- description : texte plus détaillé décrivant le produit (falcultatif)")
    st.write("- productid : identifiant unique pour le produit")
    st.write("- imageid : identifiant unique pour l’image associée au produit")
    st.write("- prdtypecode : code type du produit")
    st.write("")
    st.image("reports/figures/dataframe_info.png")

    st.write("#### 2. Statistiques descriptives")
    st.write("Les statistiques descriptives fournissent un résumé statistique des colonnes \nnumériques")
    st.dataframe(fusion.describe())
    st.write("")

    st.markdown("#### 3. Valeurs uniques de Catégorie de produit (prdtypecode)")
    st.write("On compte au total "+str(fusion['prdtypecode'].nunique())+" catégories de produits différentes, identifiées par la colonne 'prdtypecode'.")
    st.write("")

    st.write("#### 4. Vérification des doublons")
    st.write("Le jeu de donnée ne présente aucun doublon parmi les colonnes.")
    st.write("")

    st.write("#### 5. Vérification des valeurs nulles")
    st.write("Le tableau suivant nous permet de constater plus de 35% de valeurs nulles dans la colonne 'description' :")
    st.dataframe(fusion.isna().mean())

### Visualisation des données ###

if page == pages[2]:
    with open("src/streamlit/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.write("## Visualisation des données")
    st.write("")

    st.write("#### 1. Heatmap : corrélation entre les variables")
    st.image("reports/figures/heatmap.png")
    st.write("**Conclusion :** on ne retrouve aucune corrélation intéressante entre les variables mis à part entre productid et imageid qui ne semble pas être significatif pour le projet.")
    st.write("")
    st.write("#### 2. Histogramme avec estimation de la densité : prdtypecode")
    st.image("reports/figures/histogramme_avec_estimation_densite.png")
    st.write("**Conclusion :** on constate que les valeurs de codes type produit se répartissent sur 3 plages de valeurs principales (ex : entre 0 et 50, entre 1000 et 1500 et entre 2000 et 2900).")
    st.write("")

    st.write("#### 3. Histogramme : prdtypecode")
    st.image("reports/figures/histogramme.png")
    st.write("**Conclusion :** étant donnée la représentation des 27 catégories de produits, ordonnées par ordre décroissant, on constate que la catégorie 2583 se détache fortement en terme de sur-représentation et que les catégories 2905, 60, 2220, 1301, 1940 et 1180 se détachent fortement en termes de sous-représentation. On peut donc en déduire qu’il s’agit d’un problème de classification multiclasses sur des données déséquilibrées, sur lequel on s'intéressera donc principalement au f1-score.")
    st.write("")

    st.write("#### 4. Nuage de point : productid et prdtypecode")
    st.image("reports/figures/scatterplot.png")
    st.write("**Conclusion :** on remarque que les codes type se répartissent par plage discrètes de valeurs, d’où la répartition en lignes. Dans la majorité des cas, pour chaque code type, les codes produits s’étendent sur l’ensemble de la plage des productid.")

### Pre-processing ###

if page == pages[3]:
    st.write("## Pre-processing")
    st.write("")
    st.write("Rakuten met à disposition un jeu de données et fournit les images et les fichiers CSV pour X_train, y_train et X_test (sans les données y_test correspondantes). Nous avons donc décidé de fusionner X_train et y_train pour partir d’un jeu de données complet, composé de 84916 produits.")
    st.write("")

    st.write("#### 1. Descriptions des produits")
    st.write("")
    st.write("**Gestion des valeurs nulles**")
    st.write("On remarque un total de 29800 lignes sur 84916, soit plus de 35%, où la description est manquante. Pour éviter de les supprimer et donc de pénaliser notre jeu de donnée, nous avons décidé de fusionner les colonnes 'designation' et 'description' en une nouvelle colonne 'descriptif'.")
    st.write("")

    st.write("**Traitement naturel du langage (NLP)**")
    st.write("Nous avons utilisé la librairie NLTK et les méthodes NLP pour nettoyer les données textuelles :")
    st.write("- Suppression des espaces excessifs, des balises HTML, des nombres et caractères spéciaux")
    st.write("- Suppresion des mots vides et des mots de moins de 4 lettres (ex : les, tes, ces, etc …)")
    st.write("- Lemmatisation et tokenisation du texte")
    st.write("- Vectorisation pour transformer le texte en valeur numérique")
    st.write("")
    df = pd.read_csv('data//df_tokenized.csv', index_col=0)
    st.dataframe(df.head()[['descriptif', 'tokens']])
    st.write("")

    st.write("**Résultats obtenus**")
    st.write("L’étape du traitement du langage nous a permis de passer d’une liste de mots de 222906 à 137099 après nettoyage. On obtient alors 'descriptif' comme variable explicative et 'prdtypecode' comme variable cible. Aussi, pour mieux comprendre la variable cible, nous avons réalisé une étude sur les mots les plus fréquents présents par catégorie :")
    st.image("reports/figures/features_importance.png")
    st.write("Nous avons utilisé la librairie langdetect pour détecter la langue de chaque description, où on retrouve des langues prédominantes tel que le français et l'anglais soit plus de 88% du corpus de texte.")
    st.image("reports/figures/lang_detect.png")
    st.write("")
    st.write("Afin de mieux comprendre les codes types, nous avons réalisé des nuages de mots afin de se faire une idée plus claire sur le contenu des catégories. Voici un exemple de la classe 2583 qui est majoritaire :")
    st.image("reports/figures/nuage_de_mot/2583.png")
    st.write("")

    st.write("#### 2. Images des produits")
    st.write("")
    st.write("Les images fournient par Rakuten en 500x500 pixels et en mode RGB, ont été redimensionné en 125x125 pixels en les passant en nuance de gris. Nous avons ensuite crée un nouveau dataframe avec les chemins d'accès, faisant référence aux images, pour variable explicative et leur label associé pour variable cible :")
    df = get_dataframe_image(X_train, Y_train, 'data/images/image_train_preprocessed/')
    st.dataframe(df.head())

### Classification des textes ###
    
if page == pages[4]:
    with open("src/streamlit/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.write("## Classification des textes")
    st.write("")

    st.write("#### 1. Optimisation des hyperparamètres")
    st.write("Nous avons utilisé 2 approches : GridSearchCV et BayesSearchCV")
    st.write("- **GridSearchCV** : effectue une recherche exhaustive en testant toutes les combinaisons possibles de paramètres dans les plages spécifiées.")
    st.write("- **BayesSearchCV** : utilise l'optimisation bayésienne, qui apprend au fur et à mesure des essais pour cibler les zones prometteuses de l'espace des hyperparamètres.")
    st.write("")

    st.write("#### 2. Ré-échantillonnage des données")
    st.write("Les données étant déséquilibrées, nous avons appliqué les techniques de ré-échantillonnage suivantes :")
    st.write("- **Sous-échantillonnage** : les méthodes RandomUnderSampler et EditedNearestNeighbour")
    st.write("- **Sur-échantillonnage** : les méthodes SMOTE et ADASYN")
    st.write("Dans les 2 cas, les scores obtenus restaient inférieurs à ceux obtenus sans ré-échantillonnage.")
    st.write("**Conclusion :** les méthodes de ré-échantillonnages ne s'avèrent pas efficaces dans notre contexte et ne seront donc pas retenues pour l'entraînement des différents modèles.")    
    st.write("")

    st.write("#### 3. Application des modèles de classification")
    choix_texte = ['LogisticRegression', 'MultinomialNB', 'ComplementNB', 'LinearSVM', 'SGDClassifier', 'DecisionTreeClassifier']
    option_texte = st.selectbox('Choix du modèle', choix_texte)
    display_texte = st.radio('Que souhaitez-vous montrer sur la partie texte ?', ('Scores de performance', 'Matrice de confusion'))
    model, model_name = prediction(option_texte)
    if display_texte == 'Scores de performance':
        accuracy, f1 = scores_texte(model, display_texte, X_test_texte, y_test_texte)
        st.write("Accuracy : ", accuracy)
        st.write("F1 Score : ", f1)
    elif display_texte == 'Matrice de confusion':
        if (model_name !='logistic_regression') : model_name+="_grid"
        st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")
    st.write("")

    st.write("#### 4. Algorithmes d'optimisation")
    st.write("Sur la base des 2 meilleurs modèles (LinearSVM et SGDClassifier), nous avons utilisé les 2 algorithmes d'optimisation suivants pour améliorer les scores de nos modèles :") 
    st.write("- AdaBoostClassifier")
    st.write("- BaggingClassifier")
    st.write("Dans les 2 cas, les scores obtenus restaient inférieurs à ceux obtenus sans optimisation.")
    st.write("Nous avons aussi tenté d'optimiser les hyperparamètres sur ces algorithmes (via GridSearchCV) : les temps d'exécution se sont avérés excessivement longs.")
    st.write("")

    st.write("#### 5. Modèle retenu")
    st.write("Le modèle le plus performant est LinearSVM sans ré-échantillonnage ni optimisation avec les paramètres et résultats suivants :")
    st.write("- **Paramètres** : C=0.7399651076649312, max_iter=10000")
    model, model_name = prediction('LinearSVM')
    accuracy, f1 = scores_texte(model, 'Scores de performance', X_test_texte, y_test_texte)
    st.write("- **Accuracy** : "+str(accuracy))
    st.write("- **F1-score** : "+str(f1))
    st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")

### Classification des images ###
        
if page == pages[5]:
    st.write("## Classification des images")
    st.write("La partie classification des images a été réalisé sur un ensemble de 84916 images (67932 pour l'ensemble d'entraînement et 16984 pour l'ensemble de test). Les images étant au format .jpg, en nuance de gris et de dimension 125x125 pixels suite au pre-processing.")
    
    st.write("#### 1. Présentation des modèles")
    st.write("Nous avons utilisé deux modèles de Deep Learning de type CNN :")
    st.write("- Le modèle Sequential qui est le modèle de base de Keras")
    st.write("- Le modèle ResNet50 qui est un modèle pré-entraîné sur ImageNet")
    st.image("reports/figures/resnet50_representation.png")

    st.write("#### 2. Entraînement des modèles")
    st.write("Les deux modèles ont été entraînés sur 20 époques avec une fonction callback ReduceLROnPlateau qui permet de réduire le taux d'apprentissage lorsque la fonction de perte cesse de diminuer au bout d'un certain nombre d'époque.")
    st.write("Afin d'éviter un sur-apprentissage des données, nous avons réalisé deux types de data augmentation sur les ensembles d'entraînement et de test pour comparer les résultats obtenus.")
    st.write("- Faible augmentation des données : retournement horizontal et zoom")
    st.write("- Forte augmentation des données : retournement horizontal, zoom, rotation, décalage en hauteur et décalage en largeur")

    st.write("#### 3. Résultats obtenus")
    choix_image = ['Sequential (faible augmentation des données)', 'ResNet50 (forte augmentation des données)']
    option_image = st.selectbox('Choix du modèle', choix_image)
    
    display_texte = st.radio('Que souhaitez-vous montrer sur la partie image ?', ('Scores de performance', 'Matrice de confusion', 'Fonction de perte et score de performance en fonction du nombre d\'époque'))

    if option_image == 'Sequential (faible augmentation des données)':
        model_name = 'sequential'
    elif option_image == 'ResNet50 (forte augmentation des données)':
        model_name = 'resnet50'

    if display_texte == 'Scores de performance':
        loss, accuracy, f1 = scores_image(model_name)
        st.write("- Loss : ", round(loss, 4))
        st.write("- Accuracy : ", round(accuracy, 4))
        st.write("- F1-score : ", round(f1.mean(), 4))
    elif display_texte == 'Fonction de perte et score de performance en fonction du nombre d\'époque':
        st.image(f"reports/figures/{model_name}/plot_accuracy_and_loss.png")
    elif display_texte == 'Matrice de confusion':
        st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")

    st.write("#### 4. Modèle retenu")
    st.write("Le modèle Sequential avec faible augmentation de donnée a été retenu pour la classification des images avec des scores (accuracy=0.4636 et f1_score=0.3583) proches de ceux proposés par Rakuten, tout en évitant un sur-apprentissage du modèle.")

### Interprétation des résultats ###
        
if page == pages[6]:
    st.write("## Interprétation des résultats")
    st.write("")

    st.write("Nous avons utilisé la librairie LIME pour interpréter les résultats du modèle LinearSVM afin de mieux comprendre comment il réalise des prédictions sur notre corpus de texte pour en déduire à quelle classe appartient un produit e-commerce. Voici un exemple :")
    st.write("")
    st.image("reports/figures/lime_example.png")
    st.write("On remarque que l'exemple ci-dessus a de très faible probabilité pour la classe 40 mais a une probabilité de 0.84 d'appartenir à la classe 1280, représenté par le nuage de mots suivant :")
    st.image("reports/figures/nuage_de_mot/1280.png")
    st.write("")
    st.write("L'interprétation des modèles de Deep Learning n'a pas été réalisée pour la partie image, mais il serait intéressant d'utiliser la librairie Grad-CAM afin de mieux comprendre leur fonctionnement.")

### Conclusion ###
            
if page == pages[7]:
    st.markdown("## Conclusion")
    st.write("")

    st.write("#### 1. Synthèse")
    st.write("Le pre-processing aura été une des parties les plus cruciales pour préparer nos données, que ce soit pour la partie texte ou image, afin qu’elles soient exploitables par nos modèles. Elle nous a également permis de réduire considérablement le temps d’entraînement de nos modèles et d’augmenter leur performance.") 
    st.write("Nous avons expérimenté divers modèles de Machine Learning pour le texte et de Deep Learning pour les images, retenant finalement les modèles LinearSVM et Sequential en raison de leurs performances.")    
    st.write("La classification des produits à l'aide de leur description et de leur designation peut être assurée par le modèle LinearSVM qui permet d'obtenir plus de 80% de bonnes prédictions. En revanche, pour ce qui est des images, nos modèles ne permettent pas de classer correctement des produits avec moins de 50% de bonnes prédictions, et ceux malgré le fait qu’ils soient proches de ceux proposés par Rakuten. ")
    st.write("")

    st.write("#### 2. Perspectives")
    st.write("Pour la partie texte, il serait intéressant de tester de nouveaux modèles, plus adapté au corpus de texte tel que BERT ou des réseaux de neurones de type CNN afin de comparer les performances obtenues.")
    st.write("Concernant la partie image, la partie pre-processing devrait certainement être améliorée afin de rendre le jeu de données plus exploitable pour nos modèles. Nous pourrions également mettre en place de nouveaux modèles de Deep Learning (ex : RNN, TNN ou d’autres modèles Hugging Face).")
    st.write("Enfin, nous n’avons pas eu le temps de mettre en place la partie Multimodal Learning pour regrouper nos scores (texte et image) malgré nos recherches sur des modèles multimodaux à double encodeur ou à encodeur commun. Cette dernière partie permettrait d'associer le texte et l’image pour la prédiction des produits e-commerce de Rakuten.")
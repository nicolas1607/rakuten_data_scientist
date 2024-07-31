import os
import sys
import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

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
        'LinearSVC': 'linear_svm',
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
    "Interprétabilité", # simplisse
    "Conclusion" # riadh
]

page = st.sidebar.radio("", pages)
st.sidebar.warning("Cohorte : Bootcamp DS de mai 2024\n\n**Nicolas Mormiche**\nhttps://linkedin.com/in/mormichen\n\n**Riadh Zidi**\nhttps://linkedin.com/in/riadh\n\n**Simplice Lolo Mvoumbi**\nhttps://www.linkedin.com/in/simplice-lolo-mvoumbi-726606286/\n\n**Slimane Chelouah**\nhttp://www.linkedin.com/in/slimane-chelouah")

### Introduction ###

if page == pages[0]:
    st.markdown("## Introduction")

### Exploration des données ###    

if page == pages[1]:
    st.markdown("## Exploration des données")
    visualisation_options = [
        "Sélectionner une visualisation",
        "Heatmap des corrélations",
        "Histogramme avec estimation de la densité de prdtypecode",
        "Histogramme de prdtypecode",
        "Nuage de points entre productid et prdtypecode",
    ]

    choice = st.selectbox("Choisir une visualisation à afficher", visualisation_options)
    if st.button('Afficher la visualisation'):
        if choice == "Heatmap des corrélations":
            st.image("reports/figures/heatmap.png", caption='Corrélation entre toutes les variables quantitatives')
        elif choice == "Histogramme avec estimation de la densité de prdtypecode":
            st.image("reports/figures/histogramme_avec_estimation_densite.png", caption='Répartition des valeurs de prdtypecode avec estimation de la densité')
        elif choice == "Histogramme de prdtypecode":
            st.image("reports/figures/histogramme.png", caption='Répartition des valeurs de prdtypecode')
        elif choice == "Nuage de points entre productid et prdtypecode":
            st.image("reports/figures/scatterplot.png", caption='Catégorie du produit en fonction du productid')

### Visualisation des données ###

if page == pages[2]:
    with open("src/streamlit/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    title1_text = "Visualisation des données"
    st.markdown(f'<h1 class="custom-title-h1">{title1_text}</h1>', unsafe_allow_html=True)
    
    # Fig1 : Heatmap : corrélation entre les variables quantitatives
    title2_text = "(Fig.1) Heatmap : corrélation entre les variables quantitatives"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    fusion = fusion.drop(['description'], axis=1)
    fusion = fusion.drop(['designation'], axis=1)
    
    size_x, size_y= 1, 1
    font_size = 3
    fig, ax = plt.subplots(figsize = (size_x, size_y))
    sns.heatmap(fusion.corr(), ax = ax, cmap = "coolwarm", annot=True, annot_kws={"size": font_size})
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    st.write(fig)
    body_text = "<b>Conclusion :</b> on ne retrouve aucune corrélation intéressante entre les variables quantitatives mis à part entre productid et imageid qui ne semble pas être significatif pour le projet."
    body_text += "<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    # Fig2 : Histogramme avec estimation de la densité : prdtypecode
    title2_text = "(Fig.2) Histogramme avec estimation de la densité : prdtypecode"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    fig = sns.displot(fusion.prdtypecode, bins=20, kde = True, rug=True, color="red")
    #plt.title('Répartition des valeurs de prdtypecode avec estimation de la densité')
    st.pyplot(fig)
    body_text = "<b>Conclusion :</b> on constate que les valeurs de codes type produit se répartissent sur 3 plages de valeurs principales (ex : entre 0 et 50, entre 1000 et 1500 et entre 2000 et 2900)."
    body_text += "<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    
    # Fig3 : Histogramme : prdtypecode
    title2_text = "(Fig.3) Histogramme : prdtypecode"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    distribution = fusion['prdtypecode'].value_counts()
    distribution_df = distribution.reset_index()
    distribution_df.columns = ['prdtypecode', 'count']

    # Tracer le graphique à barres
    size_x, size_y= 3, 3
    font_size = 5
    fig, ax = plt.subplots(figsize = (size_x, size_y))
    distribution_df.plot(kind='bar', x='prdtypecode', y='count', ax=ax)
    ax.set_xlabel('prdtypecode', fontsize=font_size)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    #ax.set_title('Répartition des valeurs de prdtypecode')
    st.pyplot(fig)
    body_text = "<b>Conclusion :</b> étant donnée la représentation des 27 catégories de produits, ordonnées par ordre décroissant, on constate que la catégorie 2583 se détache fortement en terme de sur-représentation et que les catégories 2905, 60, 2220, 1301, 1940 et 1180 se détachent fortement en termes de sous-représentation. <b>On peut donc en déduire qu’il s’agit d’un problème de classification multiclasses sur des données déséquilibrées.</b>"
    body_text += "<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    
    
    # Fig4 : Nuage de point : productid et prdtypecode
    title2_text = "(Fig.4) Nuage de point : productid et prdtypecode"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    fig = plt.figure(figsize = (size_x, size_y))
    fig = sns.relplot(x=fusion.productid, y=fusion.prdtypecode)
    #plt.title('Catégorie du produit en fonction du productid')
    st.pyplot(fig)
    body_text = "<b>Conclusion :</b> on remarque que les codes type se répartissent par plage discrètes de valeurs, d’où la répartition en lignes. Dans la majorité des cas, pour chaque code type, les codes produits s’étendent sur l’ensemble de la plage des productid."
    body_text += "<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)

### Pre-processing ###

if page == pages[3]:
    st.write("## Pre-processing")
    st.write("La partie pre-processing a été réalisée indépendamment sur un ensemble de 84916 descriptions d'un côté et 84916 images de l'autre. Ces données représentent le jeu d'entraînement fournis par Rakuten qui nous servira de dataset de base pour la modélisation.")

    st.write("### 1. Descriptions des produits")
    st.write("- **Gestion des valeurs nulles**")
    st.code("fusion.isna().mean()")
    st.dataframe(fusion.isna().mean())
    st.write("On remarque un total de 29800 lignes sur 84916, soit plus de 35%, où la description est manquante. Pour éviter de les supprimer et donc de pénaliser notre jeu de donnée, nous avons décidé de fusionner les colonnes 'designation' et 'description' en une nouvelle colonne 'descriptif'.")

    st.write("- **Traitement naturel du langage (NLP)**")
    st.write("Nous avons utilisé la librairie NLTK et les méthodes NLP pour nettoyer les données textuelles :")
    st.code("# Retrait des espaces excessifs\ntexte = re.sub('\s{2,}', ' ', texte)\n\n# Mettre en minuscule\ntexte = texte.lower()\n\n# Supprimer les balises HTML\ntexte = BeautifulSoup(texte, 'html.parser').get_text()\n\n# Supprimer les nombres\ntexte = re.sub('\d+', '', texte)\n\n# Supprimer les accents\ntexte = unidecode(texte)\n\n# Supprimer les caractères spéciaux\ntexte = re.sub('[^a-zA-Z]', ' ', texte)\n\n# Supprimer les stopwords et les mots de moins de 4 lettres\nstop_words = set(stopwords.words())\ntexte = ' '.join([word for word in texte.split() if word not in stop_words])\ntexte = ' '.join([word for word in texte.split() if len(word) > 3])")
    st.write("Une lemmatisation et une tokenisation ont été réalisées pour affiner l'analyse du corpus de texte :")
    st.code("# Lemmatisation\nlemmatizer = WordNetLemmatizer()\ndf['descriptif_cleaned'] = df['descriptif_cleaned'].progress_apply(lambda texte: ' '.join([lemmatizer.lemmatize(mot) for mot in texte.split()]))\n\n# Tokenisation\ndf['tokens'] = df['descriptif_cleaned'].progress_apply(word_tokenize)")
    st.write("Ce qui nous permet d'obtenir un corpus de texte nettoyé, qui sera ensuite transformer en vecteurs numériques à l'aide de la méthode TfidfVectorizer pour les rendre exploitables par les modèles de machine learning.")   
    df = pd.read_csv('data//df_tokenized.csv', index_col=0)
    st.dataframe(df.head()[['descriptif', 'tokens']])

    st.write("- **Résultats obtenus**")
    st.write("L’étape du traitement du langage nous a permis de passer d’une liste de mots de 222906 à 137099 après nettoyage. Aussi, pour mieux comprendre la variable cible, nous avons réalisé une étude sur les mots les plus fréquents présents par catégorie. ")
    st.image("reports/figures/features_importance.png")
    st.write("Nous avons utilisé la librairie langdetect pour détecter la langue de chaque description, où on retrouve des langues prédominantes tel que le français et l'anglais soit plus de 88% du corpus de texte.")
    st.image("reports/figures/lang_detect.png")
    st.write("On remarque la présence de 9 modalités où la langue n'est pas reconnue. En effet, une fois la colonne 'descriptif' prétraitée à l'aide des méthodes NLP, aucun mot n'a été retenu ce qui semble indiqué que le commerçant n'a pas renseigné de description et de désignation avec des mots suffisamment explicite.")
    st.write("Afin de mieux comprendre les codes types, nous avons réalisé des nuages de mots afin de se faire une idée plus claire sur le contenu des catégories. Voici un exemple de la classe 2583 qui est majoritaire :")
    st.image("reports/figures/nuage_de_mot/2583.png")

    st.write("### 2. Images des produits")
    st.write("Les images ont été redimensionnées en 125x125 pixels en les passant en nuance de gris :")
    st.code("input_path = 'data/images/image_train/'\noutput_path = 'data/images/image_train_preprocessed/'\n\nimage = cv2.imread(input_path+filename, cv2.IMREAD_COLOR)\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\nimage = cv2.resize(image, (125, 125))\n\ncv2.imwrite(output_path+filename, image)\n")
    st.write("Nous avons ensuite crée un nouveau dataframe avec les chemins des images et leur label associé :")
    st.code("df = X.merge(y, left_index=True, right_index=True)\ndf['filepath'] = df.apply(lambda row: output_path + 'image_' + str(row['imageid']) + '_product_' + str(row['productid']) + '.jpg', axis=1)\ndf['prdtypecode'] = df['prdtypecode'].astype(str)\ndf = df[['filepath', 'prdtypecode']]\ndf.head()")
    df = get_dataframe_image(X_train, Y_train, 'data/images/image_train_preprocessed/')
    st.dataframe(df.head())

### Classification des textes ###
    
if page == pages[4]:
    with open("src/streamlit/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    title1_text = "Classification des textes"
    st.markdown(f'<h1 class="custom-title-h1">{title1_text}</h1>', unsafe_allow_html=True)
    #st.markdown("## Classification des produits : texte")
    
    
    title2_text = "Stratégies de classification"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    body_text = "Nous avons écarté les stratégies de classification OneVSRest et OneVsOne, souvent utilisées dans les problèmes multi-classe. En effet, ces stratégies ne sont pas adaptées aux grands ensembles de données."
    body_text +="<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    
    title2_text = "Optimisation des hyperparamètres"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    body_text = "Nous avons utilisé 2 approches : GridSearchCV et BayesSearchCV<br>"
    body_text += "- <b>GridSearchCV</b> : effectue une recherche exhaustive en testant toutes les combinaisons possibles de paramètres dans les plages spécifiées.<br>"
    body_text += "- <b>BayesSearchCV</b> : utilise l'optimisation bayésienne, qui apprend au fur et à mesure des essais pour cibler les zones prometteuses de l'espace des hyperparamètres."
    body_text +="<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    
    title2_text = "Ré-échantillonnage des données"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    body_text = "Les données étant déséquilibrées, nous avons appliqué les techniques de ré-échantillonnage suivantes :<br>"
    body_text += "- <b>Sous-échantillonnage</b> : les méthodes RandomUnderSampler et EditedNearestNeighbour<br>"
    body_text += "- <b>Sur-échantillonnage</b> : les méthodes SMOTE et ADASYN<br>"
    body_text +="<br>"
    body_text +="Dans les 2 cas, les scores obtenus restaient inférieurs à ceux obtenus sans ré-échantillonnage."
    body_text +="<br><br>"
    body_text += "<b>Conclusion :</b> les méthodes de ré-échantillonnages ne s'avèrent pas efficaces dans notre contexte et ne seront donc pas retenues pour l'entraînement des différents modèles."
    body_text +="<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)
    
    
    title2_text = "Application de modèles de Classification de machine learning"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    choix_texte = ['LogisticRegression', 'MultinomialNB', 'ComplementNB', 'LinearSVC', 'SGDClassifier', 'DecisionTreeClassifier']
    option_texte = st.selectbox('Choix du modèle', choix_texte)
    display_texte = st.radio('Que souhaitez-vous montrer sur la partie texte ?', ('Scores de performance', 'Matrice de confusion'))
   
    model, model_name = prediction(option_texte)
    #st.write(model, model_name)
    if display_texte == 'Scores de performance':
        #st.write(model, display_texte, X_test_texte, y_test_texte)
        accuracy, f1 = scores_texte(model, display_texte, X_test_texte, y_test_texte)
        #body_text = "Accuracy : "+ str(accuracy)+"<br>"
        #body_text += "F1 Score : "+ str(f1)+"<br>"
        #st.markdown(f'<div class="custom-body-red">{body_text}</div>', unsafe_allow_html=True)
        st.write("Accuracy : ", accuracy)
        st.write("F1 Score : ", f1)
    elif display_texte == 'Matrice de confusion':
        if (model_name !='logistic_regression') : model_name+="_grid"
        #st.write("matrice_confusion_heatmap_"+model_name+".png")
        st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")

    
    title2_text = "Algorithmes d'optimisation"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    body_text = "Sur la base des 2 meilleurs modèles (LinearSVM et SGDClassifier), nous avons utilisé les 2 algorithmes d'optimisation suivants pour améliorer les scores de nos modèles :<br>" 
    body_text +="- AdaBoostClassifier<br>"
    body_text +="- BaggingClassifier<br>"
    body_text +="Dans les 2 cas, les scores obtenus restaient inférieurs à ceux obtenus sans optimisation.<br>"
    body_text +="Nous avons aussi étudié la mise en place d'hyperparamètres sur ces algorithmes (via GridSearchCV) : les temps d'exécution se sont avérés excessivement longs."
    body_text +="<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)

    #st.write("## Titre2")
    #st.write("corps du texte.")

    title2_text = "Modèle retenu"
    st.markdown(f'<h2 class="custom-title-h2">{title2_text}</h2>', unsafe_allow_html=True)
    body_text = "Le modèle le plus performant est LinearSVM sans ré-échantillonnage ni optimisation avec les paramètres et résultats suivants :<br>"
    body_text +="- <b>Paramètres</b> : C=0.7399651076649312, max_iter=10000<br>"
    #body_text +="- <b>Accuracy</b> : 0.8193, F1-score : 0.8171<br>"
    model, model_name = prediction('LinearSVC')
    accuracy, f1 = scores_texte(model, 'Scores de performance', X_test_texte, y_test_texte)
    body_text +="- <b>Accuracy</b> : "+str(accuracy)+", <b>F1-score</b> : "+str(f1)+"<br>"
    body_text +="<br><br>"
    st.markdown(f'<div class="custom-body">{body_text}</div>', unsafe_allow_html=True)

    #"Scores de performance"
    #'Matrice de confusion':
    st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")

### Classification des images ###
        
if page == pages[5]:
    st.write("## Classification des images")
    st.write("La partie classification des images a été réalisé sur un ensemble de 84916 images (67932 pour l'ensemble d'entraînement et 16984 pour l'ensemble de test). Les images étant au format .jpg, en nuance de gris et de dimension 125x125 pixels suite au pre-processing.")
    
    st.write("#### 1. Présentation des modèles")
    st.write("Pour se faire, nous avons utilisé deux modèles de Deep Learning de type CNN avec pour paramètres :")
    st.code("num_classes = 27\nepochs = 20\nbatch_size = 8\nlearning_rate = 0.001\npatience = 4\nmin_lr=0.0001\n")
    st.write("- Le modèle Sequential qui est le modèle de base de Keras")
    st.code("model = Sequential()\nmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Conv2D(128, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(512, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))")
    st.write("- Le modèle ResNet50 qui est un modèle pré-entraîné sur ImageNet")
    st.code("base_model = ResNet50(\n    weights='imagenet', \n    include_top=False, \n    input_shape=(size, size, 3), \n    pooling='avg', \n    classes=num_classes, \n    classifier_activation='softmax', \n    input_tensor=None\n)\nx = base_model.output\nx = Dense(128, activation='relu')(x)\noutputs = Dense(num_classes, activation='softmax')(x)\nmodel = Model(inputs=base_model.input, outputs=outputs)")
    
    st.write("#### 2. Entraînement des modèles")
    st.write("Les deux modèles ont été entraînés avec une fonction callback ReduceLROnPlateau pour réduire le taux d'apprentissage lorsque la fonction de perte cesse de diminuer au bout d'un certain nombre d'époque.")
    st.code("reduce_lr_callback = ReduceLROnPlateau(\n    monitor='val_loss', \n    factor=0.2, \n    patience=patience, \n    min_lr=min_lr\n)")
    st.write("Afin d'éviter un sur-apprentissage des données, nous avons réalisé deux types de data augmentation sur les ensembles d'entraînement et de test pour comparer les résultats obtenus.")
    st.write("- Faible augmentation des données :")
    st.code("train_datagen = ImageDataGenerator(\n    rescale=1./255,\n    horizontal_flip=True,\n    zoom_range=0.2\n)\ntest_datagen = ImageDataGenerator(rescale=1./255)")
    st.write("- Forte augmentation des données :")
    st.code("train_datagen = ImageDataGenerator(\n    rescale=0.1,\n    zoom_range=0.1,\n    rotation_range=10,\n    width_shift_range=0.1,\n    height_shift_range=0.1,\n    horizontal_flip=True\n)\ntest_datagen = ImageDataGenerator(rescale=0.1)")

    st.write("#### 3. Résultats obtenus")
    choix_image = ['Sequential (faible augmentation des données)', 'ResNet50 (forte augmentation des données)']
    option_image = st.selectbox('Choix du modèle', choix_image)
    
    display_texte = st.radio('Que souhaitez-vous montrer sur la partie texte ?', ('Scores de performance', 'Matrice de confusion', 'Fonction de perte et score de performance en fonction du nombre d\'époque'))

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

### Interprétabilité ###
        
if page == pages[6]:
    st.write("")

### Conclusion ###
            
if page == pages[7]:
    st.markdown("## Conclusion")
    st.write("Le projet a mis en lumière l'importance de la classification multiclasse et multimodale (texte et image) pour atteindre les objectifs du Challenge Rakuten. Le pre-processing s'est avéré crucial pour optimiser les données, réduisant ainsi le temps d’entraînement des modèles et améliorant leurs performances. Nous avons expérimenté divers modèles de Machine Learning pour le texte et de Deep Learning pour les images, optant finalement pour LinearSVM et Resnet50 en raison de leurs bonnes performances. Bien que des améliorations soient possibles, notamment dans le traitement des images et l'intégration de méthodes multimodales, les résultats obtenus fournissent une base solide pour de futurs développements et optimisations.")

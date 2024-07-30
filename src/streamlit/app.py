import os
import pickle
import pandas as pd
import streamlit as st

from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.model_res_net_50 import data_augmentation
from src.models.preprocessing import pre_processing_texte, pre_processing_image


import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data


def load_data():
    if not os.path.exists('data/texte_preprocessed'):
        X_train_texte, X_test_texte, y_train_texte, y_test_texte, vectorizer, df = pre_processing_texte(isResampling=False)
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
    
def scores_image(model, model_name, choice):
    if choice == 'Scores de performance':
        X_train_image_list = X_train_image['filepath'].tolist() if isinstance(X_train_image, pd.DataFrame) else X_train_image
        y_train_image_list = y_train_image['prdtypecode'].astype(str).tolist() if isinstance(y_train_image, pd.DataFrame) else y_train_image
        X_test_image_list = X_test_image['filepath'].tolist() if isinstance(X_test_image, pd.DataFrame) else X_test_image
        y_test_image_list = y_test_image['prdtypecode'].astype(str).tolist() if isinstance(y_test_image, pd.DataFrame) else y_test_image

        train_df = pd.DataFrame({'filepath': X_train_image_list, 'prdtypecode': y_train_image_list})
        test_df = pd.DataFrame({'filepath': X_test_image_list, 'prdtypecode': y_test_image_list})
        
        train_generator, test_generator = data_augmentation(train_df, test_df, 8, 125)
        
        return model.evaluate(test_generator)
    elif choice == 'Matrice de confusion':
        # return pickle.load(open(f"models/{model_name}_predictions.pkl", "rb"))
        return confusion_matrix(y_test_image, model.predict(X_test_image))

X_train_texte, X_test_texte, y_train_texte, y_test_texte, X_train_image, X_test_image, y_train_image, y_test_image = load_data()


# Importer les fichiers CSV
X_train = pd.read_csv('data/X_train.csv', index_col=0)
Y_train = pd.read_csv('data/Y_train.csv', index_col=0)

# Fusionner X_train (variables explicatives) et y_train (variable cible)
fusion = pd.merge(X_train, Y_train, left_index=True, right_index=True)


### Organisation du Streamlit ###

st.title("Challenge Rakuten")
st.sidebar.title("Challenge Rakuten")

pages = [
    "Introduction", # simplice
    "Exploration des données", # riadh
    "Data visualisation", # slimane
    "Pre-processing", #nicolas
    "Modélisation (texte)", # slimane
    "Modélisation (image)", # nicolas
    "Interprétabilité", # simplisse
    "Conclusion" # riadh
]

page = st.sidebar.radio("", pages)

### Introduction ###

if page == pages[0]:
    st.markdown("## Introduction")

### Exploration des données ###    

if page == pages[1]:
    st.write("")

### Data visualisation ###

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
    st.write("")

### Modélisation (texte) ###
    
if page == pages[4]:
    with open("src/streamlit/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    title1_text = "Classification des produits : texte"
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
    if display_texte == 'Scores de performance':
        #st.write(model, display_texte, X_test_texte, y_test_texte)
        accuracy, f1 = scores_texte(model, display_texte, X_test_texte, y_test_texte)
        #body_text = "Accuracy : "+ str(accuracy)+"<br>"
        #body_text += "F1 Score : "+ str(f1)+"<br>"
        #st.markdown(f'<div class="custom-body-red">{body_text}</div>', unsafe_allow_html=True)
        st.write("Accuracy : ", accuracy)
        st.write("F1 Score : ", f1)
    elif display_texte == 'Matrice de confusion':
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

### Modélisation (image) ###
        
if page == pages[5]:
    st.markdown("## Classification des produits : image")
    choix_image = ['Sequential', 'ResNet50']
    option_image = st.selectbox('Choix du modèle', choix_image)

    display_image = st.radio('Que souhaitez-vous montrer sur la partie image ?', ('Scores de performance', 'Matrice de confusion'))

    model, model_name = prediction(option_image)

    if display_image == 'Scores de performance':
        loss, accuracy, f1_score = scores_image(model, model_name, display_image)
        st.write("Loss : ", loss)
        st.write("Accuracy : ", accuracy)
        st.write("F1 Score : ", f1_score)
    elif display_image == 'Matrice de confusion':
        st.image(f"reports/figures/matrice_de_confusion/matrice_confusion_heatmap_{model_name}.png")

### Interprétabilité ###
        
if page == pages[6]:
    st.write("")

### Conclusion ###
            
if page == pages[7]:
    st.write("")

import os
import pickle
import pandas as pd
import streamlit as st

from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score
from preprocessing import pre_processing_texte, pre_processing_image
from model_res_net_50 import data_augmentation

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
        'Logistic Regression': 'logistic_regression',
        'MultinomialNB': 'multinomialNB',
        'ComplementNB': 'complementNB',
        'LinearSVM': 'linear_svm',
        'SGD': 'sgd',
        'Decision Tree': 'decisionTree',
        'KNN Neighbors': 'knn_neighbors',
        'Sequential': 'sequential',
        'ResNet50': 'resnet50'
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

### Organisation du Streamlit ###

st.title("Challenge Rakuten")
st.sidebar.title("Challenge Rakuten")

pages = [
    "Introduction", # simplice
    "Exploration des données", # riadh
    "Data visualisation", # slimane
    "Pre-processing", #nicolas
    "Modélisation (texte)", # nicolas
    "Modélisation (image)", # slimane
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
    st.write("")

### Pre-processing ###

if page == pages[3]:
    st.write("")

### Modélisation (texte) ###
    
if page == pages[4]:
    st.markdown("## Classification des produits : texte")
    choix_texte = ['Logistic Regression', 'MultinomialNB', 'ComplementNB', 'LinearSVM', 'SGD', 'Decision Tree', 'KNN Neighbors']
    option_texte = st.selectbox('Choix du modèle', choix_texte)

    display_texte = st.radio('Que souhaitez-vous montrer sur la partie texte ?', ('Scores de performance', 'Matrice de confusion'))

    model, model_name = prediction(option_texte)

    if display_texte == 'Scores de performance':
        accuracy, f1 = scores_texte(model, display_texte, X_test_texte, y_test_texte)
        st.write("Accuracy : ", accuracy)
        st.write("F1 Score : ", f1)
    elif display_texte == 'Matrice de confusion':
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

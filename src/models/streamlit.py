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
st.sidebar.warning("Cohorte : Bootcamp DS de mai 2024\n\nNicolas Mormiche\nhttps://linkedin.com/in/mormichen\n\nRiadh Zidi\nhttps://linkedin.com/in/riadh\n\nSimplice Lolo Mvoumbi\nhttps://linkedin.com/in/simplice\n\nSlimane Chelouah\nhttps://linkedin.com/in/slimane")

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

    choix_texte = ['LogisticRegression', 'MultinomialNB', 'ComplementNB', 'LinearSVC', 'SGDClassifier', 'DecisionTreeClassifier']
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
    st.write("## Classification des produits : image")
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
    st.write("Pour des raisons de temps de calcul, nous avons choisi de ne pas afficher les scores de performance pour les modèles d'images. Vous pouvez cependant visualiser les résultats obtenus pour la fonction de perte et pour le score de performance en fonction du nombre d'époque.")
    choix_image = ['Sequential (faible augmentation des données)', 'ResNet50 (forte augmentation des données)']
    option_image = st.selectbox('Choix du modèle', choix_image)
    model, model_name = prediction(option_image)
    st.image(f"reports/figures/{model_name}/plot_accuracy_and_loss.png")

    st.write("#### 4. Modèle retenu")
    st.write("Le modèle Sequential avec faible augmentation de donnée a été retenu pour la classification des images avec des scores (accuracy=0.4636 et f1_score=0.3583) proches de ceux proposés par Rakuten, tout en évitant un sur-apprentissage du modèle.")

### Interprétabilité ###
        
if page == pages[6]:
    st.write("")

### Conclusion ###
            
if page == pages[7]:
    st.write("")

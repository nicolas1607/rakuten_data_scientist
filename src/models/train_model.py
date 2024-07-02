import os
import pickle
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearnex import patch_sklearn

# Intel(R) Extension for Scikit-learn
patch_sklearn()

def grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres):
    
    # Créer ou charger la grille de recherche
    filename = "models/" + model_name + "_grid.pkl"
    if os.path.exists(filename):
        grid_clf = pickle.load(open(filename, "rb"))
    else:
        start_time = time.time()
        grid_clf = GridSearchCV(estimator=model, param_grid=parametres, n_jobs=-1, cv=3, verbose=1)
        grid_clf.fit(X_train, y_train)
        end_time = time.time()
        heures, minutes, secondes = convertir_duree(end_time - start_time)
        print("Temps d'entrainement du modèle :",f"{heures} heures, {minutes} minutes, et {secondes} secondes\n")
        pickle.dump(grid_clf, open(filename, "wb"))

    # Afficher les meilleurs paramètres de la grille pour notre modèle
    print("Les meilleurs paramètres de la grille :", grid_clf.best_params_)
    #print("depth atteinte :", grid_clf.tree_.max_depth)

    # Prédiction des données et affichage des résultats
    y_pred = grid_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    confusion_heatmap(y_test, y_pred, model_name+"_grid")
    print("Score grid :", grid_clf.score(X_test, y_test))
    print("F1-score grid :", f1_score(y_test, y_pred, average='weighted'))

    return grid_clf

def bayes_search(X_train, X_test, y_train, y_test, model, model_name, parametres):

    # Créer ou charger la grille de recherche
    filename = "models/" + model_name + "_bayes.pkl"
    if os.path.exists(filename):
        bayes_clf = pickle.load(open(filename, "rb"))
    else:
        bayes_clf = BayesSearchCV(estimator=model, search_spaces=parametres, n_iter=32, cv=3, verbose=1, n_jobs=-1)
        bayes_clf.fit(X_train, y_train)
        pickle.dump(bayes_clf, open(filename, "wb"))

    # Afficher les meilleurs paramètres de la grille pour notre modèle
    print(bayes_clf.best_params_)

    # Prédiction des données et affichage des résultats
    y_pred = bayes_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    confusion_heatmap(y_test, y_pred, model_name+"_grid")
    print("Score bayes :", bayes_clf.score(X_test, y_test))
    print("F1-score bayes :", f1_score(y_test, y_pred, average='weighted'))

    return bayes_clf

def modele_regression_logistique(X_train, X_test, y_train, y_test):
    
    print("Modèlisation Logistic Regression\n")

    # Créer ou charger le modèle
    if os.path.exists("models/logistic_regression.pkl"):
        model = pickle.load(open("models/logistic_regression.pkl", "rb"))
    else:
        model = LogisticRegression(C=0.1, max_iter=10000, random_state=123)
        model.fit(X_train, y_train)
        pickle.dump(model, open("models/logistic_regression.pkl", "wb"))
    
    # Prédiction des données et affichage des résultats
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Score :", accuracy)
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
    
    return model, accuracy

def modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=False):
    
    model_name = 'multinomialNB'

    if (not booGrid):
        print("Modèlisation MultinomialNB\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = MultinomialNB()
            model.fit(X_train, y_train)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        confusion_heatmap(y_test, y_pred, model_name)
        print("Score :", model.score(X_test, y_test))
        print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
    
    else:

        print("Modèlisation MultinomialNB (gridSearch)\n")
        model = MultinomialNB()
        parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None

def modele_complementNB(X_train, X_test, y_train, y_test, booGrid=False):
    
    model_name = 'complementNB'

    if (not booGrid):
        print("Modèlisation ComplementNB\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = ComplementNB()
            model.fit(X_train, y_train)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Score :", model.score(X_test, y_test))
        print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    else:

        print("Modèlisation ComplementNB (gridSearch)\n")
        model = ComplementNB()
        parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None

def modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=False):

    model_name = 'linear_svm'

    if (not booGrid):
        print("Modèlisation Linear SVM\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = LinearSVC()
            model.fit(X_train, y_train)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Score :", model.score(X_test, y_test))
        print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
    
    else:
        
        print("Modèlisation Linear SVM (gridSearch)\n")
        model = LinearSVC()
        parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'max_iter': (1000, 10000)}
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    # Créer et entraîner un booster
    # ac = AdaBoostClassifier(estimator=linear_svm, n_estimators=400, algorithm='SAMME', random_state=123)
    # ac.fit(X_train, y_train)
    # pickle.dump(linear_svm, open("models/linear_svm_adaboost.pkl", "wb"))
    # y_pred = ac.predict(X_test)
    # print("Score boosting :", ac.score(X_test, y_test))
    # print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    # Créer et entraîner un bagging
    # bc = BaggingClassifier(estimator=linear_svm, n_estimators=400, oob_score=True, random_state=123)
    # bc.fit(X_train, y_train)
    # pickle.dump(linear_svm, open("models/linear_svm_bagging.pkl", "wb"))
    # y_pred = bc.predict(X_test)
    # print("Score bagging :", bc.score(X_test, y_test))
    # print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    return None

def modele_svm(X_train, X_test, y_train, y_test, booGrid=False):

    model_name = 'svm'

    if (not booGrid):
        print("Modèlisation SVM\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = SVC()
            model.fit(X_train, y_train)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Score :", model.score(X_test, y_test))
        print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    else:

        print("Modèlisation SVM (gridSearch)\n")
        model = SVC()
        parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'gamma': (1e-6, 1e+1, 'log-uniform'), 'degree': (1, 8)}
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None

def modele_sgd(X_train, X_test, y_train, y_test, booGrid=False):

    model_name = 'sgd'

    if (not booGrid):
        print("Modèlisation SGDClassifier\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = SGDClassifier()
            model.fit(X_train, y_train)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Score :", model.score(X_test, y_test))
        print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    else:

        print("Modèlisation SGDClassifier (gridSearch)\n")
        model = SGDClassifier()
        parametres = {'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01], 'max_iter': [1000, 2000, 3000, 5000, 10000]}
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None

def modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=False):
    
    model_name = 'decisionTree'
    
    if (not booGrid):
        print("Modélisation Arbre de Décision (hors gridSearch)\n")

        # Créer ou charger le modèle
        if os.path.exists("models/"+model_name+".pkl"):
            print("Chargement du modèle sauvegardé")
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            start_time = time.time()
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            end_time = time.time()
            heures, minutes, secondes = convertir_duree(end_time - start_time)
            print("Temps d'entrainement du modèle :",f"{heures} heures, {minutes} minutes, et {secondes} secondes\n")
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        print("depth atteinte :", model.tree_.max_depth)
        # Prédiction des données et affichage des résultats
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        confusion_heatmap(y_test, y_pred, model_name)
        print("Score :", model.score(X_test, y_test))
    else: 
        print("Modélisation Arbre de Décision (gridSearch)\n")
        model = DecisionTreeClassifier()
        parametres = {
            #'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
            'criterion': ['gini', 'entropy']
        }
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None

def confusion_heatmap(y_test, y_pred, modele_name):
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 15)) 
    sns.heatmap(conf_mat, cmap = "coolwarm", annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice de confusion :'+modele_name)
    plt.savefig("reports/figures/matrice_de_confusion/matrice_confusion_heatmap_"+modele_name+".png", bbox_inches='tight')

def convertir_duree(secondes):
    minutes, secondes = divmod(secondes, 60)
    heures, minutes = divmod(minutes, 60)
    return heures, minutes, secondes
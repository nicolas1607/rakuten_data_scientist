import os
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
        grid_clf = GridSearchCV(estimator=model, param_grid=parametres, n_jobs=-1, cv=3, verbose=1)
        grid_clf.fit(X_train, y_train)
        pickle.dump(grid_clf, open(filename, "wb"))

    # Afficher les meilleurs paramètres de la grille pour notre modèle
    print(grid_clf.best_params_)

    # Prédiction des données et affichage des résultats
    # y_pred = grid_clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score grid :", grid_clf.score(X_test, y_test))

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
    print(confusion_matrix(y_test, y_pred))
    print("Score bayes_clf :", bayes_clf.score(X_test, y_test))

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

def modele_multinomialNB(X_train, X_test, y_train, y_test):
    
    print("Modèlisation MultinomialNB\n")

    # Créer ou charger le modèle
    if os.path.exists("models/multinomialNB.pkl"):
        multinomialNB = pickle.load(open("models/multinomialNB.pkl", "rb"))
    else:
        multinomialNB = MultinomialNB(alpha=0.1, fit_prior=False, force_alpha=True)
        multinomialNB.fit(X_train, y_train)
        pickle.dump(multinomialNB, open("models/multinomialNB.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = multinomialNB.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score :", multinomialNB.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    # Créer et entraîner la grille de recherche
    # parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
    # grid_search(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', parametres)

    return None

def modele_complementNB(X_train, X_test, y_train, y_test):
    
    print("Modèlisation ComplementNB\n")

    # Créer ou charger le modèle
    if os.path.exists("models/complementNB.pkl"):
        complementNB = pickle.load(open("models/complementNB.pkl", "rb"))
    else:
        complementNB = ComplementNB(alpha=0.5, fit_prior=True, force_alpha=True)
        complementNB.fit(X_train, y_train)
        pickle.dump(complementNB, open("models/complementNB.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = complementNB.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score :", complementNB.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    # Créer et entraîner la grille de recherche
    # parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
    # grid_search(X_train, X_test, y_train, y_test, complementNB, 'complementNB', parametres)

    return None

def modele_linear_svm(X_train, X_test, y_train, y_test):

    print("Modèlisation Linear SVM\n")

    # Créer ou charger le modèle
    if os.path.exists("models/linear_svm.pkl"):
        linear_svm = pickle.load(open("models/linear_svm.pkl", "rb"))
    else:
        linear_svm = LinearSVC(C=0.739965107664931, max_iter=10000, dual='auto', random_state=123)
        linear_svm.fit(X_train, y_train)
        pickle.dump(linear_svm, open("models/linear_svm.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = linear_svm.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score :", linear_svm.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
        
    # Optimisation bayésienne des hyperparamètres
    # parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'max_iter': (1000, 10000)}
    # bayes_search(X_train, X_test, y_train, y_test, linear_svm, 'linear_svm', parametres)

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

def modele_svm(X_train, X_test, y_train, y_test):

    print("Modèlisation SVM\n")

    # Créer ou charger le modèle
    if os.path.exists("models/svm.pkl"):
        svm = pickle.load(open("models/svm.pkl", "rb"))
    else:
        svm = SVC(kernel='linear', random_state=123)
        svm.fit(X_train, y_train)
        pickle.dump(svm, open("models/svm.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = svm.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score :", svm.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
        
    # Optimisation bayésienne des hyperparamètres
    # parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'gamma': (1e-6, 1e+1, 'log-uniform'), 'degree': (1, 8)}
    # bayes_search(X_train, X_test, y_train, y_test, svm, 'svm', parametres)

    return None

def modele_sgd(X_train, X_test, y_train, y_test):

    print("Modèlisation SGD\n")

    # Créer ou charger le modèle
    if os.path.exists("models/sgd.pkl"):
        sgd = pickle.load(open("models/sgd.pkl", "rb"))
    else:
        sgd = SGDClassifier(alpha=0.0001, loss='modified_huber', max_iter= 2000, penalty='l2', random_state=123)
        sgd.fit(X_train, y_train)
        pickle.dump(sgd, open("models/sgd.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = sgd.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print("Score :", sgd.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    # Optimisation bayésienne des hyperparamètres
    # parametres = {'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01], 'max_iter': [1000, 2000, 3000, 5000, 10000]}
    # bayes_search(X_train, X_test, y_train, y_test, sgd, 'sgd', parametres)

    return None
  
def convertir_duree(secondes):
    minutes, secondes = divmod(secondes, 60)
    heures, minutes = divmod(minutes, 60)
    return heures, minutes, secondes

def modele_decisionTree(X_train, X_test, y_train, y_test):

    print("Modélisation Arbre de Décision\n")
    modele = 'decisionTree'

    # Créer ou charger le modèle
    if os.path.exists("models/decisiontree.pkl"):
        print("Chargement du modèle sauvegardé")
        dt = pickle.load(open("models/decisiontree.pkl", "rb"))
    else:
        start_time = time.time()
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        end_time = time.time()
        heures, minutes, secondes = convertir_duree(end_time - start_time)
        print("Temps d'entrainement du modèle :",f"{heures} heures, {minutes} minutes, et {secondes} secondes\n")
        pickle.dump(dt, open("models/dt.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = dt.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Score grid :", dt.score(X_test, y_test))

    return None


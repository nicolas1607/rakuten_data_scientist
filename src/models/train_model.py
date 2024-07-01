import os
import pickle

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres):
    
    # Créer ou charger la grille de recherche
    filename = "models/" + model_name + "_grid.pkl"
    if os.path.exists(filename):
        grid_clf = pickle.load(open(filename, "rb"))
    else:
        parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
        grid_clf = GridSearchCV(estimator=model, param_grid=parametres)
        grid_clf.fit(X_train, y_train)
        pickle.dump(grid_clf, open(filename, "wb"))

    # Afficher les meilleurs paramètres de la grille pour notre modèle
    print(grid_clf.best_params_)

    # Prédiction des données et affichage des résultats
    y_pred = grid_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Score grid :", grid_clf.score(X_test, y_test))

    return grid_clf

def modele_svm(X_train, X_test, y_train, y_test):

    print("Modèlisation SVM\n")

    # Créer ou charger le modèle
    if os.path.exists("models/svm.pkl"):
        svm = pickle.load(open("models/svm.pkl", "rb"))
    else:
        svm = SVC()
        svm.fit(X_train, y_train)
        pickle.dump(svm, open("models/svm.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Score grid :", svm.score(X_test, y_test))

    return None

def modele_multinomialNB(X_train, X_test, y_train, y_test):
    
    print("Modèlisation MultinomialNB\n")

    # Créer ou charger le modèle
    if os.path.exists("models/multinomialNB.pkl"):
        multinomialNB = pickle.load(open("models/multinomialNB.pkl", "rb"))
    else:
        multinomialNB = MultinomialNB()
        multinomialNB.fit(X_train, y_train)
        pickle.dump(multinomialNB, open("models/multinomialNB.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = multinomialNB.predict(X_test)
    print("Score :", multinomialNB.score(X_test, y_test))

    # Créer et entraîner la grille de recherche
    grid_search(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]})

    return None

def modele_complementNB(X_train, X_test, y_train, y_test):
        
    print("Modèlisation ComplementNB\n")

    # Créer ou charger le modèle
    if os.path.exists("models/complementNB.pkl"):
        complementNB = pickle.load(open("models/complementNB.pkl", "rb"))
    else:
        complementNB = ComplementNB()
        complementNB.fit(X_train, y_train)
        pickle.dump(complementNB, open("models/complementNB.pkl", "wb"))

    # Prédiction des données et affichage des résultats
    y_pred = complementNB.predict(X_test)
    print("Score :", complementNB.score(X_test, y_test))

    # Créer et entraîner la grille de recherche
    grid_search(X_train, X_test, y_train, y_test, complementNB, 'complementNB', {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]})

    return None


def modele_regression_logistique(X_train, X_test, y_train, y_test):
    
    # Vectoriser le texte en utilisant TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Entraîner le modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Prédiction des données et affichage des résultats
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    return model, vectorizer, accuracy


import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def modele_svm(X_train, X_test, y_train, y_test):

    print("Modèlisation SVM\n")

    # Créer et entraîner le modèle
    svm = SVC()
    svm.fit(X_train, y_train)

    # Prédiction des données de l'ensemble de test
    y_pred = svm.predict(X_test)
    pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']) # matrice de confusion
    
    # Afficher les résultats
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Score :", svm.score(X_test, y_test))

    # Créer et entraîner une grille de recherche
    parametres = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'kernel':['rbf','linear', 'poly', 'sigmoid'], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
    grid_clf = GridSearchCV(estimator=svm, param_grid=parametres)
    grille = grid_clf.fit(X_train, y_train)

    # Afficher les combinaisons possibles d'hyperparamètres et la performance moyenne 
    print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])

    # Afficher les meilleurs paramètres de la grille pour notre modèle
    print(grid_clf.best_params_)

    # Prédiction des données de l'ensemble de test
    y_pred = grid_clf.predict(X_test)
    pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']) # matrice de confusion

    return None
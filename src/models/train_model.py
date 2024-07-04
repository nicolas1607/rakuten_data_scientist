import os
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearnex import patch_sklearn

# Intel(R) Extension for Scikit-learn
patch_sklearn()

def get_predictions(X_test, y_test, model, model_name):

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    confusion_heatmap(y_test, y_pred, model_name+"_grid")
    print("Score :", model.score(X_test, y_test))
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))

    return None

def grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres):
    
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

    print("Les meilleurs paramètres de la grille :", grid_clf.best_params_)

    get_predictions(X_test, y_test, grid_clf, model_name)

    model.set_params(**grid_clf.best_params_)
    model.fit(X_train, y_train)

    return model

def bayes_search(X_train, X_test, y_train, y_test, model, model_name, parametres):


    filename = "models/" + model_name + "_bayes.pkl"
    if os.path.exists(filename):
        bayes_clf = pickle.load(open(filename, "rb"))
    else:
        start_time = time.time()
        bayes_clf = BayesSearchCV(estimator=model, search_spaces=parametres, n_iter=32, cv=3, verbose=1, n_jobs=-1)
        bayes_clf.fit(X_train, y_train)
        end_time = time.time()
        heures, minutes, secondes = convertir_duree(end_time - start_time)
        print("Temps d'entrainement du modèle :",f"{heures} heures, {minutes} minutes, et {secondes} secondes\n")
        pickle.dump(bayes_clf, open(filename, "wb"))

    print("Les meilleurs paramètres de la grille :", bayes_clf.best_params_)

    get_predictions(X_test, y_test, bayes_clf, model_name)

    model.set_params(**bayes_clf.best_params_)
    model.fit(X_train, y_train)

    return model

def boosting(X_train, X_test, y_train, y_test, model, model_name, booGrid=False):

    model_name = model_name + "_boosting"

    print("Création et entrainement du booster\n")   
    boosting = AdaBoostClassifier(estimator=model, algorithm='SAMME', random_state=123)

    if (not booGrid):
        boosting.set_params(learning_rate=0.1, n_estimators=100)
        boosting.fit(X_train, y_train)
    else:
        parametres = {
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [100, 200, 500, 1000],
        }
        boosting = grid_search(X_train, X_test, y_train, y_test, boosting, model_name, parametres)

    get_predictions(X_test, y_test, boosting, model_name)

    return None

def bagging(X_train, X_test, y_train, y_test, model, model_name, booGrid=False):
    
    model_name = model_name + "_bagging"

    print("Création et entrainement du bagging\n")   
    bagging = BaggingClassifier(estimator=model, random_state=123)

    if (not booGrid):
        bagging.set_params(n_estimators=100)
        bagging.fit(X_train, y_train)
    else:
        parametres = {
            'n_estimators': [100, 200, 500, 1000],
            'max_samples': [0.5, 1.0],
            'max_features': [0.5, 1.0],
        }
        bagging = grid_search(X_train, X_test, y_train, y_test, bagging, model_name, parametres)

    get_predictions(X_test, y_test, bagging, model_name)

    return None

def modele_regression_logistique(X_train, X_test, y_train, y_test, booGrid=False):
    
    model_name = 'regression_logistique'
    
    if (not booGrid):
        print("Modélisation regression logistique (hors gridSearch)\n")

        # Créer ou charger le modèle
        if os.path.exists("models/"+model_name+".pkl"):
            print("Chargement du modèle sauvegardé")
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            start_time = time.time()
            model = LogisticRegression()
            model.fit(X_train, y_train)
            end_time = time.time()
            heures, minutes, secondes = convertir_duree(end_time - start_time)
            print("Temps d'entrainement du modèle :",f"{heures} heures, {minutes} minutes, et {secondes} secondes\n")
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

        # Prédiction des données et affichage des résultats
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        confusion_heatmap(y_test, y_pred, model_name)
        print("Score :", model.score(X_test, y_test))
    else: 
        print("Modélisation regression_logistique (gridSearch)\n")
        model = LogisticRegression()
        parametres = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'max_iter': [100, 200, 300, 500, 1000]
            
        }
        grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return None
#def modele_regression_logistique(X_train, X_test, y_train, y_test, booGrid=False):
    
    model_name = 'regression_logistique'
    if os.path.exists("models/logistic_regression.pkl"):
        model = pickle.load(open("models/logistic_regression.pkl", "rb"))
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        pickle.dump(model, open("models/logistic_regression.pkl", "wb"))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_heatmap(y_test, y_pred, model_name)
    print("Score :", accuracy)
    print("F1-score :", f1_score(y_test, y_pred, average='weighted'))
    
    return None

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
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation MultinomialNB (gridSearch)\n")
        model = MultinomialNB()
        parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

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
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation ComplementNB (gridSearch)\n")
        model = ComplementNB()
        parametres = {'alpha':[x / 10 for x in range(1, 11, 1)], 'force_alpha':[True,False], 'fit_prior':[True,False]}
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

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
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation Linear SVM (gridSearch)\n")
        model = LinearSVC()
        parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'max_iter': (1000, 10000)}
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

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
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation SVM (gridSearch)\n")
        model = SVC()
        parametres = {'C': (1e-6, 1e+6, 'log-uniform'), 'gamma': (1e-6, 1e+1, 'log-uniform'), 'degree': (1, 8)}
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

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
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation SGDClassifier (gridSearch)\n")
        model = SGDClassifier()
        parametres = {'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01], 'max_iter': [1000, 2000, 3000, 5000, 10000]}
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

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
        get_predictions(X_test, y_test, model, model_name)
    else: 
        print("Modélisation Arbre de Décision (gridSearch)\n")
        model = DecisionTreeClassifier()
        parametres = {
            #'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
            'criterion': ['gini', 'entropy']
        }
        model = grid_search(X_train, X_test, y_train, y_test, model, model_name, parametres)

    return model

def modele_xgboost(X_train, X_test, y_train, y_test, booGrid=False):

    model_name = 'xgboost'

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    if (not booGrid):
        print("Modèlisation XGBClassifier\n")
        if os.path.exists("models/"+model_name+".pkl"):
            model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        else:
            model = XGBClassifier(learning_rate=0.1, objective='multi:softmax', num_class=27, random_state=123)
            model.fit(X_train, y_train_encoded)
            pickle.dump(model, open("models/"+model_name+".pkl", "wb"))
        get_predictions(X_test, y_test, model, model_name)
    else:
        print("Modèlisation XGBClassifier (gridSearch)\n")
        model = XGBClassifier(learning_rate=0.1, objective='multi:softmax', num_class=27, random_state=123)
        parametres = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            # 'subsample': [0.6, 0.8, 1.0],
            # 'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }
        model = bayes_search(X_train, X_test, y_train_encoded, y_test_encoded, model, model_name, parametres)

    return model

def confusion_heatmap(y_test, y_pred, modele_name):
    
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 15))
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
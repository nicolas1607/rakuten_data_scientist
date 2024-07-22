from preprocessing import pre_processing_texte, pre_processing_image
from train_model import *
from bert_model import *
from model_res_net_50 import *

# Pre-processing des données
X_train, X_test, y_train, y_test, vectorizer, df = pre_processing_texte(isResampling=False)
# X_train, X_test, y_train, y_test = pre_processing_image(size=125)

# Classification des produits (texte) : modèle retenu
linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=True)
interpretability(linearSVM, df, vectorizer)

# Classification des produits (image) : modèles retenus
# sequential = model_cnn(X_train, X_test, y_train, y_test, size=125)
# resnet50 = model_resnet50(X_train, X_test, y_train, y_test, size=125)

# Modèle texte n°1 : Logistic Regression
# logistic_regression = modele_logistic_regression(X_train, X_test, y_train, y_test, booGrid=False)
# logistic_regression = modele_logistic_regression(X_train, X_test, y_train, y_test, booGrid=True)

# Modèle texte n°2 : MultinomialNB
# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=False)
# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=True)

# Modèle texte n°3 : ComplementNB
# complementNB = modele_complementNB(X_train, X_test, y_train, y_test, booGrid=False)
# complementNB = modele_complementNB(X_train, X_test, y_train, y_test, booGrid=True)

# Modèle texte n°4 : LinearSVC (avec boosting et bagging)
# linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=False)
# linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=True)
# linearSVM_boosting = boosting(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=False)
# linearSVM_bagging = bagging(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=False)

# Modèle texte n°5 : SGD (avec boosting et bagging)
# sgd = modele_sgd(X_train, X_test, y_train, y_test, booGrid=False)
# sgd = modele_sgd(X_train, X_test, y_train, y_test, booGrid=True)
# sgd_boosting = boosting(X_train, X_test, y_train, y_test, sgd, 'sgd', booGrid=False)
# sgd_bagging = bagging(X_train, X_test, y_train, y_test, sgd, 'sgd', booGrid=False)

# Modèle texte n°6 : DecisionTreeClassifier
# modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=False)
# modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=True)

# Modèle texte n°7 : KNeighborsClassifier
# modele_knn_neighbors(X_train, X_test, y_train, y_test, booGrid=False)
# modele_knn_neighbors(X_train, X_test, y_train, y_test, booGrid=True)

# Modèle texte n°8 : SVM
# svm = modele_svm(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer

# Modèle texte n°9 : XGBoost
# modele_xgboost(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer

# Modèle texte n°10 : BERT
# X_train, X_test, y_train, y_test = pre_processing_texte(tokenizer_name='bert')
# modele_bert(X_train, X_test, y_train, y_test, booGrid=False)
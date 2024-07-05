from preprocessing import pre_processing
from train_model import *
from bert_model import *

# Pre-processing
X_train, X_test, y_train, y_test = pre_processing()

# Modélisation de base
# modele_regression_logistique(X_train, X_test, y_train, y_test, booGrid=False)
# modele_regression_logistique(X_train, X_test, y_train, y_test, booGrid=True)

# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=False)
# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=True)
# multinomialNB_boosting = boosting(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', booGrid=False)
# multinomialNB_bagging = bagging(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', booGrid=False)

# complementNB = modele_complementNB(X_train, X_test, y_train, y_test, booGrid=False)
# complementNB = modele_complementNB(X_train, X_test, y_train, y_test, booGrid=True)
# complementNB_boosting = boosting(X_train, X_test, y_train, y_test, complementNb, 'complementNb', booGrid=False)
# complementNB_bagging = bagging(X_train, X_test, y_train, y_test, complementNb, 'complementNb', booGrid=False)

# linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=False)
# linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=True)
# linearSVM_boosting = boosting(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=False)
# linearSVM_bagging = bagging(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=False)

# sgd = modele_sgd(X_train, X_test, y_train, y_test, booGrid=False)
# sgd = modele_sgd(X_train, X_test, y_train, y_test, booGrid=True)
# sgd_boosting = boosting(X_train, X_test, y_train, y_test, sgd, 'sgd', booGrid=False)
# sgd_bagging = bagging(X_train, X_test, y_train, y_test, sgd, 'sgd', booGrid=False)

# modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=False)
# modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=True)

# modele_knn_neighbors(X_train, X_test, y_train, y_test, booGrid=False)
# modele_knn_neighbors(X_train, X_test, y_train, y_test, booGrid=True)

# svm = modele_svm(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer

# modele_xgboost(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer

# Modélisation avancée
# X_train, X_test, y_train, y_test = pre_processing(tokenizer_name='bert')
# modele_bert(X_train, X_test, y_train, y_test, booGrid=False)
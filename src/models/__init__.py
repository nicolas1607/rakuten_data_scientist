from preprocessing import pre_processing
from train_model import *

# Pre-processing
X_train, X_test, y_train, y_test = pre_processing()

# Modélisation de base
# modele_regression_logistique(X_train, X_test, y_train, y_test, booGrid=False)

# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=False)
# multinomialNB = modele_multinomialNB(X_train, X_test, y_train, y_test, booGrid=True)
# multinomialNB_boosting = boosting(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', booGrid=False)
# multinomialNB_bagging = bagging(X_train, X_test, y_train, y_test, multinomialNB, 'multinomialNB', booGrid=False)

# modele_complementNB(X_train, X_test, y_train, y_test, booGrid=False)
# modele_complementNB(X_train, X_test, y_train, y_test, booGrid=True)

# modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=False)
# modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=True)

# modele_svm(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer

# modele_sgd(X_train, X_test, y_train, y_test, booGrid=False)
# modele_sgd(X_train, X_test, y_train, y_test, booGrid=True)

# modele_decisionTree(X_train, X_test, y_train, y_test, booGrid=False)

# modele_xgboost(X_train, X_test, y_train, y_test, booGrid=False)
# pas d'optimisation car trop long à executer
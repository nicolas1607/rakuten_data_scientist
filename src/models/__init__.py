from preprocessing import pre_processing
from train_model import *

# Pre-processing
X_train, X_test, y_train, y_test = pre_processing()

# Modélisation de base
modele_regression_logistique(X_train, X_test, y_train, y_test)
# modele_multinomialNB(X_train, X_test, y_train, y_test)
# modele_complementNB(X_train, X_test, y_train, y_test)
# modele_linear_svm(X_train, X_test, y_train, y_test)
# modele_svm(X_train, X_test, y_train, y_test)
# modele_sgd(X_train, X_test, y_train, y_test)
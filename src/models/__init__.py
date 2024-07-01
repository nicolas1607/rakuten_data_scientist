from preprocessing import pre_processing
from train_model import *

# Pre-processing
X_train, X_test, y_train, y_test = pre_processing()

# Modélisation de base
# modele_svm(X_train, X_test, y_train, y_test)
# modele_multinomialNB(X_train, X_test, y_train, y_test)
# modele_complementNB(X_train, X_test, y_train, y_test)

modele_decisionTree(X_train, X_test, y_train, y_test)
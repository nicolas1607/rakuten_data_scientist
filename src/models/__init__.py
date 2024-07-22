<<<<<<< HEAD
import os
import gc
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from preprocessing import *
from train_model import *
import os
import numpy as np
from PIL import Image
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
X_train, X_test, y_train, y_test = pre_processing_image(size=125)
cnn_model = model_cnn(X_train, X_test, y_train, y_test, size=125)

# num_classes = len(np.unique(y_train))
# model, history = train_evaluate_cnn(X_train, X_test, y_train, y_test, num_classes, epochs=20, batch_size=32)
#pre_processing_image()
# Pre-processing
#X_train, X_test, y_train, y_test = pre_processing()
# pre_processing_image()
# Modélisation de base
# Logistic_Regression = modele_Logistic_Regression(X_train, X_test, y_train, y_test, booGrid=False)
# Logistic_Regression = modele_Logistic_Regression(X_train, X_test, y_train, y_test, booGrid=True)
# Logistic_Regression_boosting = boosting(X_train, X_test, y_train, y_test, Logistic_Regression, 'Logistic_Regression', booGrid=False)
# Logistic_Regression_bagging = bagging(X_train, X_test, y_train, y_test, Logistic_Regression, 'Logistic_Regression', booGrid=False)

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
# linearSVM_boosting = boosting(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=True)
# linearSVM_bagging = bagging(X_train, X_test, y_train, y_test, linearSVM, 'linearSVM', booGrid=False)

=======
from preprocessing import pre_processing_texte, pre_processing_image
from train_model import *
from bert_model import *
from model_res_net_50 import *

# Pre-processing des données
X_train, X_test, y_train, y_test = pre_processing_texte(isResampling=False)
X_train, X_test, y_train, y_test = pre_processing_image(size=125)

# Classification des produits (texte) : modèles retenus
# linearSVM = modele_linear_svm(X_train, X_test, y_train, y_test, booGrid=True)
# sgd = modele_sgd(X_train, X_test, y_train, y_test, booGrid=True)

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
>>>>>>> a3c244fea75605ce23eb8d2e6795d47a7e83cd30
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
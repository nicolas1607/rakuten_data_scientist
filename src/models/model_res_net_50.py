import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import seaborn as sns
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

def data_augmentation(train_df, test_df, batch_size, size):

    train_datagen = ImageDataGenerator(
        rescale=0.1,
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=0.1)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

def get_predictions(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_class = y_pred.argmax(axis = 1)
    
    if isinstance(y_test, pd.Series):
        y_test_class = y_test.values
    else:
        y_test_class = y_test

    y_test_class = np.array(y_test_class).astype(int)
    y_pred_class = np.array(y_pred_class).astype(int)

    print(classification_report(y_test_class, y_pred_class, zero_division=0))
    print(confusion_matrix(y_test_class, y_pred_class))

def plot_loss_and_accuracy(model_history, model_name):

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(model_history['loss'])
    plt.plot(model_history['val_loss'])
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(model_history['accuracy'])
    plt.plot(model_history['val_accuracy'])
    plt.title('Model accuracy by epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.savefig("reports/figures/resnet50/"+model_name+".png", bbox_inches='tight')

def model_resnet50(X_train, X_test, y_train, y_test, size):
    
    model_name = "resnet50"

    num_classes = 27
    epochs = 20
    batch_size = 8
    learning_rate = 0.001
    patience = 4

    train_df = pd.DataFrame({'filepath': X_train, 'prdtypecode': y_train})
    test_df = pd.DataFrame({'filepath': X_test, 'prdtypecode': y_test})

    train_generator, test_generator = data_augmentation(train_df, test_df, batch_size, size)

    if os.path.exists("models/"+model_name+".pkl"):
        model = pickle.load(open("models/"+model_name+".pkl", "rb"))
        model_history = pickle.load(open("models/"+model_name+"_history.pkl", "rb"))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg', classes=num_classes, classifier_activation='softmax', input_tensor=None)

        # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, min_lr=0.001)
        model_checkpoint_callback = ModelCheckpoint(filepath='models/resnet50/checkpoint_{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, mode='min')

        x = base_model.output
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=learning_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy', tfa.metrics.F1Score(num_classes=27, name='f1_score')]
        )

        model_history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[
                # early_stopping_callback, 
                reduce_lr_callback, 
                model_checkpoint_callback
            ]
        )

        pickle.dump(model, open("models/"+model_name+".pkl", "wb"))
        pickle.dump(model_history.history, open("models/"+model_name+"_history.pkl", "wb"))

    plot_loss_and_accuracy(model_history, model_name)

    loss, accuracy, f1_score = model.evaluate(test_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}, F1 Score: {f1_score}')
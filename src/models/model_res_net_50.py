import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def plot_accuracy(model_history, model_name):
    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    # plt.ylim([0,1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model_name + ' accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig("models/"+model_name+"_accuracy.png", bbox_inches='tight')

def plot_loss(model_history, model_name):
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    # plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name + ' loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig("models/"+model_name+"_loss.png", bbox_inches='tight')

def model_resnet50(X_train, X_test, y_train, y_test, size):
    
    model_name = "resnet50"

    num_classes = 27
    epochs = 20
    batch_size = 8
    learning_rate = 0.001
    patience = 4

    # Tester les callbacks

    train_df = pd.DataFrame({'filepath': X_train, 'prdtypecode': y_train})
    test_df = pd.DataFrame({'filepath': X_test, 'prdtypecode': y_test})

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

    if os.path.exists("models/"+model_name+".pkl"):
        model = pickle.load(open("models/"+model_name+".pkl", "rb"))
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

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        model_history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[
                # early_stopping_callback, 
                reduce_lr_callback, 
                model_checkpoint_callback
            ]
            # steps_per_epoch=train_generator.n // batch_size
        )

        pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

    loss, accuracy = model.evaluate(test_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Prédictions sur le jeu de test
    # y_pred = model.predict(X_test)

    # y_pred_class = y_pred.argmax(axis = 1)
    # y_test_class = y_test.argmax(axis = 1)

    # print(classification_report(y_test_class, y_pred_class))
    # print(confusion_matrix(y_test_class, y_pred_class))

    # Graphiques
    # plot_accuracy(model_history, model_name)
    # plot_loss(model_history, model_name)
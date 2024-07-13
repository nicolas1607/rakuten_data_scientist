import os
import pickle
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

def model_resnet50(X_train, X_test, y_train, y_test, size):
    
    model_name = "resnet50"

    num_classes = 27
    epochs = 4
    batch_size = 32
    learning_rate = 0.001
    img_size = (size, size)

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_df = pd.DataFrame({'filepath': X_train, 'prdtypecode': y_train})
    test_df = pd.DataFrame({'filepath': X_test, 'prdtypecode': y_test})

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='prdtypecode',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    if os.path.exists("models/"+model_name+".pkl"):
        model = pickle.load(open("models/"+model_name+".pkl", "rb"))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(size, size, 3))

        x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        # inputs = base_model.input
        # x = Dense(128, activation='relu')(inputs)
        # x = Dense(50, activation='relu')(x)
        # outputs = Dense(num_classes, activation='softmax')(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // batch_size
        )

        pickle.dump(model, open("models/"+model_name+".pkl", "wb"))

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
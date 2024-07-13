import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

def model_resnet50(X_train, X_test, y_train, y_test, size):
    
    batch_size = 32
    num_classes = 27
    img_size = (size, size)

    # Créer les générateurs d'images
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

    # Créer le modèle ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Geler les couches du modèle
    for layer in base_model.layers:
        layer.trainable = False

    # Compiler le modèle
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size
    )

    # Évaluer le modèle
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
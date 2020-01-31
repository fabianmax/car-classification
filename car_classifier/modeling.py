import tensorflow as tf

from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class Model:

    def __init__(self, base: str, shape: tuple, classes: list):
        self.shape = shape
        self.classes = classes
        self.history = None

        if base == 'ResNet':
            self.base_model = ResNet50V2(include_top=False,
                                         input_shape=self.shape,
                                         weights='imagenet')

            self.base_model.trainable = False

            add_to_base = self.base_model.output
            add_to_base = GlobalAveragePooling2D(data_format='channels_last')(add_to_base)

        elif base == 'VGG16':
            self.base_model = VGG16(include_top=False,
                                    input_shape=self.shape,
                                    weights='imagenet')

            self.base_model.trainable = False

            add_to_base = self.base_model.output
            add_to_base = Flatten()(add_to_base)
            add_to_base = Dense(1024, activation="relu")(add_to_base)
            add_to_base = Dropout(0.25)(add_to_base)
            add_to_base = Dense(1024, activation="relu")(add_to_base)

        new_output = Dense(len(self.classes), activation="softmax")(add_to_base)
        self.model = Model(self.base_model.input, new_output)

    def compile(self):

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(0.0001),
                           metrics=["categorical_accuracy"])

    def train(self, ds_train: tf.data.Dataset, ds_valid: tf.data.Dataset, epochs):

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       restore_best_weights=True)

        callbacks = [early_stopping]

        # Fitting
        self.history = self.model.fit(ds_train,
                                      epochs=epochs,
                                      validation_data=ds_valid,
                                      callbacks=callbacks)

        return self.history


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class TransferModel:

    def __init__(self, base: str, shape: tuple, classes: list):
        """
        Class for transfer learning from either VGG16 or ResNet

        Args:
            base: String giving the name of the base model (either 'VGG16' or 'ResNet')
            shape: Input shape as tuple (height, width, channels)
            classes: List of class labels
        """
        self.shape = shape
        self.classes = classes
        self.history = None

        # Class allows for two base models (VGG16 oder ResNet)
        # Use pre-trained ResNet model
        if base == 'ResNet':
            self.base_model = ResNet50V2(include_top=False,
                                         input_shape=self.shape,
                                         weights='imagenet')

            self.base_model.trainable = False

            add_to_base = self.base_model.output
            add_to_base = GlobalAveragePooling2D(data_format='channels_last')(add_to_base)

        # Use pre-trained VGG16
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

        # Add final output layer
        new_output = Dense(len(self.classes), activation="softmax")(add_to_base)
        self.model = Model(self.base_model.input, new_output)

    def compile(self):
        """
        Compile method
        """
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(0.0001),
                           metrics=["categorical_accuracy"])

    def train(self,
              ds_train: tf.data.Dataset,
              epochs: int,
              steps_per_epoch: int = None,
              ds_valid: tf.data.Dataset = None):
        """
        Training method

        Args:
            ds_train: training data as tf.data.Dataset
            ds_valid: validation data as tf.data.Dataset
            epochs: number of epochs to train
            steps_per_epoch: Number of steps per epoch

        Returns
            Training history in self.history
        """

        # Define early stopping as callback
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       restore_best_weights=True)

        callbacks = [early_stopping]

        # Fitting
        self.history = self.model.fit(ds_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=ds_valid,
                                      callbacks=callbacks)

        return self.history

    def evaluate(self, ds_test: tf.data.Dataset):
        """
        Evaluation method

        Args:
            ds_test: Testing data as tf.data.Dataset

        Returns:
              None
        """

        # TODO add a function return
        self.model.evaluate(ds_test)

    def predict(self, ds_new: tf.data.Dataset, proba: bool = True):
        """
        Prediction method

        Args:
            ds_new: New data as tf.data.Dataset
            proba: Boolean if probabilities should be returned

        Returns:
            class labels or probabilities
        """

        p = self.model.predict(ds_new)

        if proba:
            return p
        else:
            return [np.argmax(x) for x in p]

    def plot(self, what: str = 'metric'):
        """
        Method for training/validation visualization
        Takes self.history and plots it
        """

        if self.history is None:
            AttributeError("No training history available, call TransferModel.train first")

        if what not in ['metric', 'loss']:
            AttributeError(f'type must be either "loss" or "metric"')

        if what == 'metric':
            metric = self.model.metrics_names[1]
            y_1 = self.history.history[metric]
            y_2 = self.history.history['val_' + metric]
            y_label = metric
        elif what == 'loss':
            y_1 = self.history.history['loss']
            y_2 = self.history.history['val_loss']
            y_label = 'loss'

        plt.plot(y_1)
        plt.plot(y_2)
        plt.title('Model Performance')
        plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()




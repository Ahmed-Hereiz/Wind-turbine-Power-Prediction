import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.losses import mean_squared_error
from keras import layers, metrics, Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint




class ForwardfeedNN:
    """A forward-feed neural network model for regression.

    This model implements a forward-feed neural network architecture for regression
    problems using the Keras API from TensorFlow. The architecture consists of several
    dense layers with batch normalization and dropout regularization to avoid overfitting.

    Parameters:
        None

    Attributes:
        model: A Keras Sequential model that contains the neural network architecture.

    Methods:
        _build_model: Private method that builds the neural network architecture.
        fit: Trains the model on the training data.
        predict: Generates predictions for new input data.
    """
    def __init__(self):
        
        self.model = self._build_model()
        self.filepath = 'BestNet.h5'
        self.checkpoint = ModelCheckpoint(self.filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        
    def _build_model(self):
        
        model = tf.keras.Sequential([
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=1)
        ])

        optimizer = tf.keras.optimizers.Adam()

        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[metrics.MeanAbsoluteError(name='mae')])
        
        return model

    def fit(self, X, y, epochs=200, batch_size=256*2, verbose=1):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[self.checkpoint])
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    
class TrainingHistoryPlotter:
    """
    A class to plot the accuracy and loss of a Keras model training history.
    """

    def __init__(self, history, colors=['#115ff0','#1ba2e0']):
        """
        Parameters:
        history (History): The history object returned by the `fit` method of a Keras model.
        """
        self.history = history
        self.colors = colors

    def plot(self):
        """
        Plots the accuracy and loss of the model training history.
        """
        # Plot accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        ax1.plot(self.history.history['mae'],color=self.colors[0])
        ax1.plot(self.history.history['val_mae'],color=self.colors[1])
        ax1.set_title('Model mae')
        ax1.set_ylabel('mae')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Val'], loc='upper left')

        # Plot loss
        ax2.plot(self.history.history['loss'],color=self.colors[0])
        ax2.plot(self.history.history['val_loss'],color=self.colors[1])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Val'], loc='upper left')

        # Show the plot
        plt.show()
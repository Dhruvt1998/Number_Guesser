from enum import Enum

import tensorflow as tf
import numpy as np


class ModelVariant(Enum):
    L2E5 = 1
    L1E5 = 2
    L2E3 = 3
    L1E3 = 4


def prepare_datasets():
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Binarize the training data
    for train in range(len(x_train)):
        for row in range(28):
            for x in range(28):
                if x_train[train][row][x] != 0:
                    x_train[train][row][x] = 1
    return x_train, y_train, x_test, y_test


def train_model(model_id: str, x_train: np.array, y_train: np.array, model_variant: ModelVariant):
    # Define and compile the model

    if model_variant == ModelVariant.L2E5:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=5)

    elif model_variant == ModelVariant.L1E5:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=5)

    elif model_variant == ModelVariant.L2E3:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=3)

    elif model_variant == ModelVariant.L1E3:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=3)

    else:
        print("No correct model variant")

    # Save the model
    model.save('data/model_' + model_id + '.h5')
    print("Model " + model_id + " saved")


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_datasets()
    train_model(str(100), x_train, y_train, ModelVariant.L2E5)

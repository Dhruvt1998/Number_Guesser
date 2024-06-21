import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import random

from src import model_training

matplotlib.use('tkagg')

'''
# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)
'''

# Binarize the training data
for train in range(len(x_train)):
    for row in range(28):
        for x in range(28):
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1


def evaluate(model_id: str, x_test, y_test, test_size: int, demo: bool):
    # Load the saved model without the optimizer state
    model = tf.keras.models.load_model('data/model_' + model_id + '.h5')

    # Compile the model again after loading
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Make predictions on a subset of the test data (random 10 samples for demonstration)
    random_start = random.randint(0, 89)
    predictions = model.predict(x_test[random_start:random_start + test_size])

    # Initialize a counter for incorrect predictions
    count = 0

    # Loop through the predictions and compare with actual labels
    for i in range(len(predictions)):
        guess = np.argmax(predictions[i])
        actual = y_test[random_start + i]

        print("I predict this number is a:", guess)
        print("Number Actually Is a:", actual)

        if guess != actual:
            count += 1

        if demo:
            # Display the image in a new window
            plt.figure()
            plt.imshow(x_test[random_start + i], cmap=plt.cm.binary)
            plt.title(f'Predicted: {guess}, Actual: {actual}')
            plt.show(block=False)
            plt.pause(1)

    # Calculate and print the number of wrong predictions and accuracy
    print("The program got", count, 'wrong, out of', len(x_test[:10]))
    accuracy = 100 - ((count / len(x_test[:10])) * 100)
    print(f'{accuracy}% correct')
    return accuracy


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = model_training.prepare_datasets()
    # Binarize the test data to match training preprocessing
    x_test = np.where(x_test > 0, 1, 0)
    evaluate(str(99), x_test, y_test, 10, True)

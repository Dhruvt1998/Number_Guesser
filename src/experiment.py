import pickle
import threading

import numpy as np
from matplotlib import pyplot as plt

from src.model_training import prepare_datasets, train_model, ModelVariant
from src.model_test import evaluate

avg_accuracies = []
all_accuracies = [[]]
avg_losses = []
all_losses = [[]]


def experiment(number: str, mv: ModelVariant):
    model_id = mv.name + '_' + number
    train_model(model_id, x_train, y_train, mv)
    loss, acc = evaluate(model_id, x_test, y_test)
    accuracies.append(acc)
    losses.append(loss)


def save_data():
    f = open("demofile2.txt", "a")
    for acc in avg_accuracies:
        f.write(str(acc))
    f.close()
    with open("data/experiment_data", 'wb') as f:
        pickle.dump(avg_accuracies, f)


def visualize_from_file(filename):
    with open("data/experiment_data", 'rb') as f:
        data = pickle.load(f)
        visualize_exp(filename, data)


def visualize_exp(filename, data):
    variants = ['L2E5', 'L1E5', 'L2E2', 'L1E2']
    fig, ax = plt.subplots()

    def add_labels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha='center')

    ax.set_ylabel('correct_guesses')
    ax.set_xticks([0.0, 1.0, 2.0, 3.0], variants)
    ax.set_title('Performance of models, varying in hidden layers and training epochs')
    add_labels(variants, np.round(data, 5))
    bar_colors = ['deepskyblue', 'lightskyblue', 'orange', 'gold']
    bar_labels = ['2 hidden layers, 5 epochs', '1 hidden layer, 5 epochs', '2 hidden layers, 3 epochs',
                  '1 hidden layer, 3 epochs']
    plt.ylim(0.9, 1.00)
    ax.bar(variants, data, label=bar_labels, color=bar_colors)
    #ax.plot(variants, data)
    plt.savefig('data/' + filename + '.png')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_datasets()
    threads = list()

    #Using 4 different model-variants regarding hidden layers and training-epochs
    for i in range(4):
        mv = ModelVariant(i + 1)
        accuracies = []
        losses = []

        #creating 10 models and trainign of each variant
        for j in range(50):
            x = threading.Thread(target=experiment(str(j), mv), args=(j,), daemon=True)
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()

        #saving the average Accuracy of each model variant (over all ten models trained)
        all_accuracies.append(accuracies)
        avg_accuracies.append(sum(accuracies) / len(accuracies))
        all_losses.append(losses)
        avg_losses.append(sum(losses) / len(losses))

    save_data()
    visualize_exp('direct_plot', avg_accuracies)
    #visualize_from_file('from_file')

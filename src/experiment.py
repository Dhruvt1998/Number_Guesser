import threading

from src.model_training import prepare_datasets, train_model, ModelVariant
from src.model_test import evaluate

accuracies = []


def experiment(id: str):
    train_model(id, x_train, y_train, ModelVariant.L2E5)
    accuracies.append(evaluate(id, x_test, y_test, 100, False))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_datasets()
    threads = list()

    for i in range(10):
        id = "2hidden_" + str(i)
        x = threading.Thread(target=experiment(id), args=(i,), daemon=True)
        threads.append(x)
        x.start()

    for i in range(10):
        id = "2hidden_" + str(i)
        x = threading.Thread(target=experiment(id), args=(i,), daemon=True)
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    print(accuracies)

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

from utilities import visualize_classifier


def load_arrays(filepath):
    data = np.loadtxt(filepath, delimiter=",")
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def train_classifier(x, y):
    classifier = GaussianNB()
    classifier.fit(x, y)
    return classifier


def get_prediction_accuracy(x, y, y_prediction):
    return 100.0 * (y == y_prediction).sum() / x.shape[0]


def print_accuracy(classifier, x, y, num_folds=3):
    values = cross_val_score(classifier, x, y, scoring="accuracy", cv=num_folds)
    print("Accuracy: " + str(round(100 * values.mean(), 2)) + "%")


def print_precision(classifier, x, y, num_folds=3):
    values = cross_val_score(
        classifier, x, y, scoring="precision_weighted", cv=num_folds
    )
    print("Precision: " + str(round(100 * values.mean(), 2)) + "%")


def print_recall(classifier, x, y, num_folds=3):
    values = cross_val_score(classifier, x, y, scoring="recall_weighted", cv=num_folds)
    print("Recall: " + str(round(100 * values.mean(), 2)) + "%")


def print_f1(classifier, x, y, num_folds=3):
    values = cross_val_score(classifier, x, y, scoring="f1_weighted", cv=num_folds)
    print("F1: " + str(round(100 * values.mean(), 2)) + "%")


x, y = load_arrays("Лабораторна робота 1/data_multivar_nb.txt")
classifier = train_classifier(x, y)

y_prediction = classifier.predict(x)
accuracy = get_prediction_accuracy(x, y, y_prediction)
print(f"Accuracy of Naive Bayes = {round(accuracy, 2)}%")
print_accuracy(classifier, x, y)
print_precision(classifier, x, y)
print_recall(classifier, x, y)
print_f1(classifier, x, y)


x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.2, random_state=3
)
new_classifier = train_classifier(x_training, y_training)

y_test_prediction = new_classifier.predict(x_test)
accuracy = get_prediction_accuracy(x_test, y_test, y_test_prediction)
print(f"Accuracy of the new classifier = {round(accuracy, 2)}%")

visualize_classifier(new_classifier, x_test, y_test)

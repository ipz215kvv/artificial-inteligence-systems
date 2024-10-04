import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix as get_confusion_matrix


def load_arrays(filepath):
    data = np.loadtxt(filepath, delimiter=",")
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def create_svm(x, y):
    svm = SVC()
    svm.fit(x, y)
    return svm


def create_naive(x, y):
    naive = GaussianNB()
    naive.fit(x, y)
    return naive


def predict(classifier, x):
    return classifier.predict(x)


def evaluate_model(classifier, x, y, num_folds=3):
    accuracy = cross_val_score(
        classifier, x, y, cv=num_folds, scoring="accuracy"
    ).mean()
    precision = cross_val_score(
        classifier, x, y, cv=num_folds, scoring="precision_macro"
    ).mean()

    recall = cross_val_score(
        classifier, x, y, cv=num_folds, scoring="recall_macro"
    ).mean()
    f1 = cross_val_score(classifier, x, y, cv=num_folds, scoring="f1_macro").mean()

    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1-score = {f1}")

    y_prediction = classifier.predict(x)
    confusion_matrix = get_confusion_matrix(y, y_prediction)
    print("Confusion Matrix:\n", confusion_matrix)


x, y = load_arrays("Лабораторна робота 1/data_multivar_nb.txt")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

svm = create_svm(x_train, y_train)
naive = create_naive(x_train, y_train)

svm_prediction = predict(svm, x_test)
naive_prediction = predict(naive, x_test)

print("SVM:")
evaluate_model(svm, x_test, y_test)

print("\nNaive Bayes:")
evaluate_model(naive, x_test, y_test)

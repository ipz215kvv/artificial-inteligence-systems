import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score as get_accuracy_score
from sklearn.metrics import recall_score as get_recall_score
from sklearn.metrics import precision_score as get_precision_score
from sklearn.metrics import f1_score as get_f1_score
from sklearn.metrics import roc_curve as get_roc_curve
from sklearn.metrics import roc_auc_score as get_roc_auc_score


def get_dataframe(filename):
    dataframe = pd.read_csv(filename)

    return dataframe


def binarize(y, threshold=0.5):
    y_prediction = np.where(y >= threshold, 1, 0)
    return y_prediction


def korniichuk_confusion_matrix(y, y_prediction):
    tp = korniichuk_tp(y, y_prediction)
    fn = korniichuk_fn(y, y_prediction)
    fp = korniichuk_fp(y, y_prediction)
    tn = korniichuk_tn(y, y_prediction)
    return np.array([[tn, fp], [fn, tp]])


def korniichuk_tp(y, y_prediction):
    result = 0
    for actual, predicted in zip(y, y_prediction):
        if actual == 1 and predicted == 1:
            result += 1
    return result


def korniichuk_fn(y, y_prediction):
    result = 0
    for actual, predicted in zip(y, y_prediction):
        if actual == 1 and predicted == 0:
            result += 1
    return result


def korniichuk_fp(y, y_prediction):
    result = 0
    for actual, predicted in zip(y, y_prediction):
        if actual == 0 and predicted == 1:
            result += 1
    return result


def korniichuk_tn(y, y_prediction):
    result = 0
    for actual, predicted in zip(y, y_prediction):
        if actual == 0 and predicted == 0:
            result += 1
    return result


def split_confusion_matrix(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]
    return tp, fn, fp, tn


def print_confusion_matrix(matrix):
    tp, fn, fp, tn = split_confusion_matrix(matrix)
    print(f"TP: {tp}")
    print(f"FN: {fn}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")


def korniichuk_accuracy_score(confusion_matrix):
    tp, fn, fp, tn = split_confusion_matrix(confusion_matrix)
    return (tp + tn) / (tp + tn + fp + fn)


def korniichuk_recall_score(confusion_matrix):
    tp, fn, _, _ = split_confusion_matrix(confusion_matrix)
    return tp / (tp + fn)


def korniichuk_precision_score(confusion_matrix):
    tp, _, fp, _ = split_confusion_matrix(confusion_matrix)
    return tp / (tp + fp)


def korniichuk_f1_score(confusion_matrix):
    recall_score = korniichuk_recall_score(confusion_matrix)
    precision_score = korniichuk_precision_score(confusion_matrix)
    return (2 * (precision_score * recall_score)) / (precision_score + recall_score)


def build_roc_plot(fpr_y1, tpr_y1, fpr_y2, tpr_y2):
    plt.plot(fpr_y1, tpr_y1, "r-", label="RF")
    plt.plot(fpr_y2, tpr_y2, "b-", label="LR")
    plt.plot([0, 1], [0, 1], "k-", label="random")
    plt.plot([0, 0, 1, 1], [0, 1, 1, 1], "g-", label="perfect")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


dataframe = get_dataframe("Лабораторна робота 1/data_metrics.csv")

y, y1_prediction, y2_prediction = (
    dataframe.actual_label.array,
    dataframe.model_RF.array,
    dataframe.model_LR.array,
)
y1_prediction = binarize(y1_prediction)

# CONFUSION_MATRIX
my_confusion_matrix = korniichuk_confusion_matrix(
    y,
    y1_prediction,
)
confusion_matrix = get_confusion_matrix(y, y1_prediction)

print("Korniichuk Confusion matrix:")
print_confusion_matrix(my_confusion_matrix)

print("\nSKLearn Confusion matrix:")
print_confusion_matrix(confusion_matrix)

assert np.array_equal(confusion_matrix, my_confusion_matrix)

# ACCURACY_SCORE
my_accuracy_score = korniichuk_accuracy_score(confusion_matrix)
print(f"\nKorniichuk accuracy score = {my_accuracy_score}")

accuracy_score = get_accuracy_score(y, y1_prediction)
print(f"SKLearn accuracy score = {accuracy_score}")

assert my_accuracy_score == accuracy_score

# RECALL_SCORE
my_recall_score = korniichuk_recall_score(confusion_matrix)
print(f"\nKorniichuk recall score = {my_recall_score}")

recall_score = get_recall_score(y, y1_prediction)
print(f"SKLearn recall score = {recall_score}")

assert my_recall_score == recall_score

# PRECISION_SCORE
my_precision_score = korniichuk_precision_score(confusion_matrix)
print(f"\nKorniichuk precision score = {my_precision_score}")

precision_score = get_precision_score(y, y1_prediction)
print(f"SKLearn precision score = {precision_score}")

assert my_precision_score == precision_score

# F1_SCORE
my_f1_score = korniichuk_f1_score(confusion_matrix)
print(f"\nKorniichuk f1 score = {my_f1_score}")

f1_score = get_f1_score(y, y1_prediction)
print(f"SKLearn f1 score = {f1_score}")

assert my_f1_score == f1_score

# ROC_CURVE
fpr_y1, tpr_y1, thresholds_y1 = get_roc_curve(y, y1_prediction)
fpr_y2, tpr_y2, thresholds_y2 = get_roc_curve(y, y2_prediction)
build_roc_plot(fpr_y1, tpr_y1, fpr_y2, tpr_y2)


# ROC_AUC_SCORE
roc_auc_score1 = get_roc_auc_score(y, y1_prediction)
print(f"\nAUC 1 = {roc_auc_score1}")
roc_auc_score2 = get_roc_auc_score(y, y2_prediction)
print(f"AUC 2 = {roc_auc_score2}")

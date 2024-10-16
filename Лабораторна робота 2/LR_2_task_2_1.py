import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix as get_confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def has_missing_values(row, missing_value_symbol, missing_value_columns):
    for index in missing_value_columns:
        if row[index] == missing_value_symbol:
            return True
    return False


def read_file_to_array(
    filename, max_datapoints, missing_value_symbol="?", missing_value_columns=(1, 6, 13)
):
    data = []

    data1_points = 0
    data2_points = 0

    with open(filename, "r") as file:
        for line in file.readlines():
            if data1_points >= max_datapoints and data2_points >= max_datapoints:
                break

            row = line[:-1].split(", ")

            if has_missing_values(
                row,
                missing_value_symbol=missing_value_symbol,
                missing_value_columns=missing_value_columns,
            ):
                continue

            is_data1_point = row[-1] == ">50K" and data1_points < max_datapoints
            is_data2_point = row[-1] == "<=50K" and data2_points < max_datapoints
            if is_data1_point:
                data1_points += 1
                data.append(row)
            elif is_data2_point:
                data2_points += 1
                data.append(row)

    data = np.array(data)
    return data


def encode_labels(
    array, encoders=None, categorical_columns=(1, 3, 5, 6, 7, 8, 9, 13, 14)
):
    encoders = {}
    encoded_array = np.empty(array.shape)
    for index in categorical_columns:
        if index not in encoders.keys():
            encoder_instance = preprocessing.LabelEncoder()
            encoders[index] = encoder_instance
        encoded_array[:, index] = encoders[index].fit_transform(array[:, index])

    return encoded_array.astype(int), encoders


def split_array(array):
    x = array[:, :-1]
    y = array[:, -1]
    return x, y


def train_svm(x, y):
    classifier = OneVsOneClassifier(SVC(kernel="poly", degree=8))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=5
    )
    classifier.fit(x_train, y_train)
    return classifier, x_test, y_test


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

    print(f"Accuracy = {accuracy:.2f}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print(f"F1-score = {f1:.2f}")

    y_prediction = classifier.predict(x)
    confusion_matrix = get_confusion_matrix(y, y_prediction)
    print("Confusion Matrix:\n", confusion_matrix)


data = read_file_to_array("Лабораторна робота 2/income_data.txt", max_datapoints=25_000)
data, encoders = encode_labels(data)
x, y = split_array(data)
svm, x_test, y_test = train_svm(x, y)

input_array = [
    "37",
    "Private",
    "215646",
    "HS-grad",
    "9",
    "Never-married",
    "Handlers-cleaners",
    "Not-in-family",
    "White",
    "Male",
    "0",
    "0",
    "40",
    "United-States",
]
print(f"Input: {input_array}")
test_array = np.array([input_array])

test_array, _ = encode_labels(
    test_array, encoders=encoders, categorical_columns=(1, 3, 5, 6, 7, 8, 9, 13)
)

prediction = np.array([svm.predict(test_array)])
prediction = encoders[14].inverse_transform(prediction)[0]

print(f"Prediction: {prediction}\n")

evaluate_model(svm, x_test, y_test)

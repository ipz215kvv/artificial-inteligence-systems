import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


input_file = "Лабораторна робота 5/data_random_forests.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

# Розбиття даних на три класи на підставі міток
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Визначення сітки значень параметрів
params_grid = [
    {"n_estimators": [100], "max_depth": [2, 4, 7, 12, 16]},
    {"n_estimators": [25, 50, 100, 250], "max_depth": [4]},
]
metrics = ["precision_weighted", "recall_weighted"]

for m in metrics:
    print(f"\n##### Searching optimal parameters for {m}")
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0), params_grid, cv=5, scoring=m
    )
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    for i, params in enumerate(classifier.cv_results_["params"]):
        avg_score = classifier.cv_results_["mean_test_score"][i]
        print(params, "-->", round(avg_score, 3))

    print("\nBest parameters:", classifier.best_params_)
    y_pred = classifier.predict(X_test)
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))

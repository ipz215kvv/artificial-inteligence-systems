import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from LR_4_task_5 import generate_data


def plot_learning_curves(model, X_train, y_train, X_test, Y_test):
    train_errors = []
    test_errors = []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(sm.mean_squared_error(y_train_predict, y_train[:m]))
        test_errors.append(sm.mean_squared_error(y_test_predict, y_test))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="test")
    plt.show()


X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

lin_reg = linear_model.LinearRegression()
plot_learning_curves(lin_reg, X_train, y_train, X_test, y_test)

polynomial_10_degree_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", linear_model.LinearRegression()),
    ]
)
plot_learning_curves(polynomial_10_degree_regression, X_train, y_train, X_test, y_test)

polynomial_2_degree_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin_reg", linear_model.LinearRegression()),
    ]
)
plot_learning_curves(polynomial_2_degree_regression, X_train, y_train, X_test, y_test)

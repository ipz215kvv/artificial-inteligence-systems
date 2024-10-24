import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures


def parse_data(filename):
    data = np.loadtxt(filename, delimiter=",")

    X, y = data[:, :-1], data[:, -1]
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training

    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    return X_train, y_train, X_test, y_test


def evaluate_performance(y, y_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y, y_pred), 2))
    print(
        "Median absolute error =",
        round(sm.median_absolute_error(y, y_pred), 2),
    )
    print(
        "Explain variance score =",
        round(sm.explained_variance_score(y, y_pred), 2),
    )
    print("R2 score =", round(sm.r2_score(y, y_pred), 2))


X_train, y_train, X_test, y_test = parse_data(
    "Лабораторна робота 4/data_multivar_regr.txt"
)

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)

print("Linear regressor performance:")
evaluate_performance(y_test, y_test_pred)

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

print("\nLinear regression:\n", regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

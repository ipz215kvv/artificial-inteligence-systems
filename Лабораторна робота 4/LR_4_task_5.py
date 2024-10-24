import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


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


def show_plot(y, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Виміряно")
    ax.set_ylabel("Передбачено")
    plt.show()


def generate_data():
    m = 100
    X = 6 * np.random.rand(m, 1) - 4
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    return X, y


if __name__ == "__main__":
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    X_train_transformed = polynomial.fit_transform(X_train)

    model = linear_model.LinearRegression()
    model.fit(X_train_transformed, y_train)

    X_test_transformed = polynomial.fit_transform(X_test)
    y_test_predict = model.predict(X_test_transformed)

    print("Polynomial сoefficient:\n", model.coef_, model.intercept_)
    print("\nPolynomial regressor performance:")
    evaluate_performance(y_test, y_test_predict)
    show_plot(y_test, y_test_predict)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris


iris = load_iris()
X = iris["data"]
y = iris["target"]

# Створюємо модель
kmeans = KMeans(
    n_clusters=8,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=0.0001,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
)
# Навчаємо
kmeans.fit(X)

# Прогнозуємо кластери
y_kmeans = kmeans.predict(X)

# Візуалізуємо результати
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)


# Функція для пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # Визначаємо підходящі кластери
        labels = pairwise_distances_argmin(X, centers)

        # Визначаємо центри
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Зупиняємось, якщо центри не змінні
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


# Кластеризація
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# Кластеризація зі зміненим random_state
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# Кластеризація з 3 кластерами
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

plt.show()

import json
import numpy as np
import pandas as pd
from sklearn import covariance, cluster
import yfinance as yf  # Заміна `quotes_historical_yahoo_ochl`


# Вхідний файл із символічними позначеннями компаній
input_file = "Лабораторна робота 7/company_symbol_mapping.json"  # Завантажено з https://github.com/PacktPublishing/Artificial-Intelligence-with-Python/blob/master/Chapter%2004/code/company_symbol_mapping.json

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, "r") as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = "2003-07-03"
end_date = "2007-05-04"
quotes = []
for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date)
    if not data.empty:
        # Обчислення різниці між двома видами котирувань
        data["Price_Difference"] = data["Close"] - data["Open"]

        symbol_quotes = data[["Price_Difference"]].reset_index()
        symbol_quotes["Ticker"] = symbol

        quotes.append(symbol_quotes)

quotes = pd.concat(quotes, ignore_index=True)

pivot_df = quotes.pivot(index="Date", columns="Ticker", values="Price_Difference")

# Fill any missing values with zeros (or another method, if needed)
pivot_df.fillna(0, inplace=True)

# Extract the data as a 2D NumPy array
quotes_diff = pivot_df.values

# Нормалізація
X = quotes_diff.copy()
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()


# Виведення результатів
for i in range(num_labels + 1):
    cluster_symbols = [symbols[j] for j in range(len(labels)) if labels[j] == i]
    print("Cluster", i + 1, "==>", ", ".join(cluster_symbols))

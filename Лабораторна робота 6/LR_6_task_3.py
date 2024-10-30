import warnings
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
# Читаємо файл
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

# Викидуємо пропущені значення
data = data.dropna(subset=["price"])

# Категоризуємо дані для роботи з класифікатором
data["price_category"] = pd.cut(
    data["price"],
    bins=[0, 30, 60, 100, float("inf")],
    labels=["low", "medium", "high", "very_high"],
)

# Викидуємо не категоризовані дані
data.drop(columns=["price", "insert_date", "start_date", "end_date"], inplace=True)

# Перетворення рядкових даних на числові
label_encoders = {}
for column in [
    "price_category",
    "origin",
    "destination",
    "train_type",
    "train_class",
    "fare",
]:
    instance = LabelEncoder()
    data[column] = instance.fit_transform(data[column])
    label_encoders[column] = instance

# Визначаємо ознаки та мітки
X = data.drop("price_category", axis=1)
y = data["price_category"]

# Створення даних для навчання і тестування
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Навчаємо класифікатор
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Прогнозування на одиночному прикладі
datapoint = X_test.iloc[0]

decoded_datapoint = datapoint.copy()
for column in label_encoders:
    if column == "price_category":
        continue

    decoded_datapoint[column] = label_encoders[column].inverse_transform(
        [datapoint[column]]
    )[0]

print(f"Datapoint:\n{decoded_datapoint}")

prediction = classifier.predict([datapoint])
decoded_prediction = label_encoders["price_category"].inverse_transform(prediction)
print("\nPrice prediction:", decoded_prediction[0])

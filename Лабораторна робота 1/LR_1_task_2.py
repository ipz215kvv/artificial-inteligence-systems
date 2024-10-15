import numpy as np
from sklearn import preprocessing


def binarize(target, threshold=2.1):
    binarizer = preprocessing.Binarizer(threshold=threshold)
    return binarizer.transform(target)


def print_mean_and_deviation(data):
    print(f"Mean: {data.mean(axis=0)}")
    print(f"Std deviation: {data.std(axis=0)}")


def scale(target):
    return preprocessing.scale(target)


def min_max_scale(target):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(target)


def normalize(target, norm="l1"):
    return preprocessing.normalize(target, norm=norm)


input_data = np.array(
    [
        [-2.3, -1.6, -6.1],
        [-2.4, -1.2, 4.3],
        [3.2, 3.1, 6.1],
        [-4.4, 1.4, -1.2],
    ]
)

binarized_data = binarize(input_data, 2.1)
print(f"Binarized data:\n{binarized_data}")

print()

print("BEFORE:")
print_mean_and_deviation(input_data)
scaled_data = scale(input_data)
print("AFTER:")
print_mean_and_deviation(scaled_data)

print()

min_max_scaled_data = min_max_scale(input_data)
print(f"Min max scaled data:\n{min_max_scaled_data}")

print()

l1_normalized_data = normalize(input_data, "l1")
print(f"l1 normalized data:\n{l1_normalized_data}")

print()

l2_normalized_data = normalize(input_data, "l2")
print(f"l2 normalized data:\n{l2_normalized_data}")

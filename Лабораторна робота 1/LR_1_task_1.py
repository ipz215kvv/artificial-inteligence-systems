from sklearn import preprocessing


def train_encoder(target):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(target)

    return encoder


def print_label_mapping(encoder):
    print("Label mapping:")
    for i, item in enumerate(encoder.classes_):
        print(f"{item} --> {i}")


def encode_labels(encoder, labels):
    return encoder.transform(labels)


def print_encoded_labels(input, output):
    print(f"Labels: {input}")
    print(f"Encoded values: {list(output)}")


def decode_values(encoder, values):
    return encoder.inverse_transform(values)


def print_decoded_values(input, output):
    print(f"Values: {input}")
    print(f"Decoded labels: {list(output)}")


input_labels = ["red", "black", "red", "green", "black", "yellow", "white"]

encoder = train_encoder(input_labels)
print_label_mapping(encoder)
print()

labels = ["green", "red", "black"]
encoded_values = encode_labels(encoder, labels)
print_encoded_labels(labels, encoded_values)
print()

values = [3, 0, 4, 1]
decoded_labels = decode_values(encoder, values)
print_decoded_values(values, decoded_labels)

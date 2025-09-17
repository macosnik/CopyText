import numpy as np
import csv
from model import Net

def load_dataset(path):
    data = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = np.array(data, dtype=float)
    X = data[:, :-1]
    y_raw = data[:, -1]
    return X, y_raw

def encode_labels(y_raw):
    classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[c] for c in y_raw], dtype=int)
    return y_encoded, classes

X, y_raw = load_dataset("dataset.csv")
y, classes = encode_labels(y_raw)

net = Net([X.shape[1], 5, 5, len(classes)])
net.train(X, y, epochs=1000, lr=0.01)
net.save("model.json")

nn = Net.load("model.json")
preds = nn.predict(X[:100])
y = y[:100]

print(f"Верно: {sum([1 if preds[i] == y[i] else 0 for i in range(len(y))])}/{len(y)}")

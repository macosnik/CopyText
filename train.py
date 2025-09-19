import numpy as np
import csv
from model import Net

def load_dataset(path):
    with open(path, "r") as f:
        data = np.array(list(csv.reader(f)), dtype=float)
    return data[:, :-1], data[:, -1]

def encode_labels(y):
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[c] for c in y], int), classes

if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")
    y, classes = encode_labels(y_raw)

    net = Net([X.shape[1], 6, 6, len(classes)])
    net.train(X, y, 10000, 0.01)
    net.save("model.json")

    nn = Net.load("model.json")
    preds, y_true = nn.predict(X), y

    print(f"Верно: {(preds == y_true).sum()}/{len(y_true)}")

import os, csv, random
from model import Net

def load_dataset(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
    X = [[float(v) for v in row[:-1]] for row in data]
    y = [row[-1] for row in data]
    return X, y

def shuffle_dataset(X, y):
    idx = list(range(len(y)))
    random.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]

def result(net):
    preds = net.predict(X)
    correct = sum(int(p == yi) for p, yi in zip(preds, y))
    print(f"\nТочность на обучающем наборе: {correct}/{len(y)} = {correct / len(y):.2%}")

if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")
    classes = sorted(set(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = [class_to_idx[c] for c in y_raw]

    X, y = shuffle_dataset(X, y)

    net = Net([len(X[0]), 128, len(classes)])

    net.train(X, y, epochs=100, lr=0.1)
    result(net)

    net.train(X, y, epochs=500, lr=0.1)
    result(net)

    net.train(X, y, epochs=1000, lr=0.1)
    result(net)

    net.save("model.json")

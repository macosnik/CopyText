from model import Net
import csv, random, os

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
    X_shuf = [X[i] for i in idx]
    y_shuf = [y[i] for i in idx]
    return X_shuf, y_shuf

if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")
    classes = sorted(set(y_raw))

    os.makedirs("models", exist_ok=True)
    [os.remove(os.path.join("models", f)) for f in os.listdir("models")]

    for cls in classes:
        base = cls.split("-")[0]  # базовая группа
        X_filtered, y_filtered = [], []

        for xi, yi in zip(X, y_raw):
            if yi == cls:
                X_filtered.append(xi)
                y_filtered.append(1)
            else:
                if yi.split("-")[0] != base:
                    X_filtered.append(xi)
                    y_filtered.append(0)

        X_shuf, y_shuf = shuffle_dataset(X_filtered, y_filtered)

        net = Net([len(X[0]), 3, 2])
        net.train(X_shuf, y_shuf, epochs=10000, lr=0.1)

        path = os.path.join("models", f"model_{cls}.json")
        net.save(path)

        nn = Net.load(path)
        preds = nn.predict(X_shuf)

        total = sum(1 for v in y_shuf if v == 1)
        correct = sum(1 for p, v in zip(preds, y_shuf) if p == 1 and v == 1)

        print(f"\nВерно для '{cls}': {correct}/{total}\n")


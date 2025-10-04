import csv, random
from model import Net

EXCEPTIONS = {
    "0": ["О", "о"],
    "О": ["0", "о"],
    "о": ["0", "О"],
    "3": ["З", "з"],
    "З": ["3", "з"],
    "з": ["3", "З"],
}

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

def build_classes(y_raw):
    classes = []
    for c in y_raw:
        if c in classes:
            continue

        opposite = None
        if c.isalpha():
            if c.islower():
                opposite = c.upper()
            else:
                opposite = c.lower()

        excluded = EXCEPTIONS.get(c, [])

        classes.append(c)

        if opposite and opposite in y_raw and opposite not in excluded:
            pass

    return sorted(classes)

def result(net, X, y):
    preds = net.predict(X)
    correct = sum(int(p == yi) for p, yi in zip(preds, y))
    print(f"\nТочность на обучающем наборе: {correct}/{len(y)} = {correct / len(y):.2%}")

def per_class_stats(net, X, y, classes):
    preds = net.predict(X)
    total = len(y)
    print("\nСтатистика по меткам:")
    for idx, cls in enumerate(classes):
        mask = [yi == idx for yi in y]
        n = sum(mask)
        if n == 0:
            continue
        correct = sum(int(p == yi) for p, yi, m in zip(preds, y, mask) if m)
        acc = correct / n
        print(f"  {cls}: {correct}/{n} = {acc:.2%}")

if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")

    classes = build_classes(y_raw)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = [class_to_idx[c] for c in y_raw if c in class_to_idx]
    X = [xi for xi, yi in zip(X, y_raw) if yi in class_to_idx]

    X, y = shuffle_dataset(X, y)

    net = Net([len(X[0]), 128, len(classes)], dropout=0.1, classes=classes)

    for epochs in range(50):
        net.train(X, y, epochs=100, lr=0.1)
        result(net, X, y)
        net.save("model.json")

    per_class_stats(net, X, y, classes)

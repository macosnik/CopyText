import os, csv, random
from collections import defaultdict
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


if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")
    classes = sorted(set(y_raw))

    os.makedirs("models", exist_ok=True)
    for f in os.listdir("models"):
        os.remove(os.path.join("models", f))

    by_class = defaultdict(list)
    for i, yi in enumerate(y_raw):
        by_class[yi].append(i)

    for cls in classes:
        base = cls.split("-")[0]
        pos_idx = by_class[cls]
        neg_idx = [i for i, yi in enumerate(y_raw) if yi.split("-")[0] != base]

        neg_by_class = defaultdict(list)
        for i in neg_idx:
            neg_by_class[y_raw[i]].append(i)

        per_class = max(1, 1000 // len(neg_by_class))
        neg_sample = []
        for idxs in neg_by_class.values():
            random.shuffle(idxs)
            neg_sample.extend(idxs[:per_class])

        idx_stage1 = pos_idx + neg_sample
        X1 = [X[i] for i in idx_stage1]
        y1 = [1 if y_raw[i] == cls else 0 for i in idx_stage1]
        X1, y1 = shuffle_dataset(X1, y1)

        net = Net([len(X[0]), 5, 2])
        net.train(X1, y1, epochs=1000, lr=0.1)
        print()

        idx_stage2 = pos_idx + neg_idx
        X2 = [X[i] for i in idx_stage2]
        y2 = [1 if y_raw[i] == cls else 0 for i in idx_stage2]
        X2, y2 = shuffle_dataset(X2, y2)

        net.train(X2, y2, epochs=10000, lr=0.05)

        path = os.path.join("models", f"model_{cls}.json")
        net.save(path)

        nn = Net.load(path)
        preds = nn.predict(X2)

        total_pos = sum(1 for v in y2 if v == 1)
        correct_pos = sum(1 for p, v in zip(preds, y2) if p == 1 and v == 1)

        total_neg = sum(1 for v in y2 if v == 0)
        correct_neg = sum(1 for p, v in zip(preds, y2) if p == 0 and v == 0)

        print(f"\nВерно для '{cls}':")
        print(f"  Положительные: {correct_pos}/{total_pos}")
        print(f"  Ложные (отрицательные): {correct_neg}/{total_neg}\n")


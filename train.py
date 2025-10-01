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

def evaluate(y_true, y_pred, cls):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    total_pos = sum(1 for yt in y_true if yt == 1)
    total_neg = sum(1 for yt in y_true if yt == 0)

    print()
    print(f"Метка: {cls}")
    print(f"Положительные верно: {tp}/{total_pos}")
    print(f"Отрицательные верно: {tn}/{total_neg}")
    print("="*50)

def sample_negatives(neg_idx, y_raw, k):
    neg_by_class = defaultdict(list)
    for i in neg_idx:
        neg_by_class[y_raw[i]].append(i)
    per_class = max(1, k // len(neg_by_class))
    neg_sample = []
    for idxs in neg_by_class.values():
        random.shuffle(idxs)
        neg_sample.extend(idxs[:per_class])
    return neg_sample[:k]

if __name__ == "__main__":
    X, y_raw = load_dataset("dataset.csv")
    classes = sorted(set(y_raw))

    os.makedirs("models", exist_ok=True)
    for f in os.listdir("models"):
        os.remove(os.path.join("models", f))

    by_class = defaultdict(list)
    for i, yi in enumerate(y_raw):
        by_class[yi].append(i)

    stages = [50, 100, 500, 1000]

    for cls in classes:
        pos_idx = by_class[cls]
        neg_idx = [i for i, yi in enumerate(y_raw) if yi != cls]

        net = Net([len(X[0]), 5, 2])

        # Этапы 1–4: равномерные выборки
        for k in stages:
            neg_sample = sample_negatives(neg_idx, y_raw, k)
            idx_stage = pos_idx + neg_sample
            Xs = [X[i] for i in idx_stage]
            ys = [1 if y_raw[i] == cls else 0 for i in idx_stage]
            Xs, ys = shuffle_dataset(Xs, ys)
            net.train(Xs, ys, epochs=1000, lr=0.1)
            print(f" - {cls}: обучено на {len(Xs)} примерах (отрицательных {k})")

        # Этап 5: 1000 самых трудных отрицательных
        preds = net.predict([X[i] for i in neg_idx])
        scored = [(i, (p[1] if isinstance(p, (list, tuple)) else p)) for i, p in zip(neg_idx, preds)]
        scored.sort(key=lambda x: x[1], reverse=True)
        hard_neg = [i for i, _ in scored[:1000]]
        idx_stage = pos_idx + hard_neg
        Xs = [X[i] for i in idx_stage]
        ys = [1 if y_raw[i] == cls else 0 for i in idx_stage]
        Xs, ys = shuffle_dataset(Xs, ys)
        net.train(Xs, ys, epochs=5000, lr=0.05)
        print(f" - {cls}: обучено на {len(Xs)} примерах (1000 трудных отрицательных)")

        # Этап 6: все отрицательные
        idx_stage = pos_idx + neg_idx
        Xs = [X[i] for i in idx_stage]
        ys = [1 if y_raw[i] == cls else 0 for i in idx_stage]
        Xs, ys = shuffle_dataset(Xs, ys)
        net.train(Xs, ys, epochs=1000, lr=0.01)
        print(f"{cls}: обучено на {len(Xs)} примерах (все отрицательные)")

        # Сохраняем модель
        path = os.path.join("models", f"model_{cls}.json")
        net.save(path)

        # Оценка
        preds = net.predict(X)
        y_true = [1 if yi == cls else 0 for yi in y_raw]
        y_pred = []
        for p in preds:
            if isinstance(p, (list, tuple)):
                y_pred.append(1 if p[1] > 0.5 else 0)
            else:
                y_pred.append(1 if p > 0.5 else 0)
        evaluate(y_true, y_pred, cls)

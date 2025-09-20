import csv

FILE = 'dataset.csv'
SIZE = 20

def clean(file, thr=0.0):
    with open(file) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        data = list(reader)
    new_data = []
    for r in data:
        values = [float(x) for x in r[:-1]]
        if not all(v <= thr for v in values):
            new_data.append(r)
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(new_data)

def block(v):
    g = int(float(v) * 23) + 232
    return f"\033[48;5;{g}m   \033[0m"

if __name__ == "__main__":
    clean(FILE)

    with open(FILE) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        data = list(reader)

    labels = sorted({row[-1] for row in data})
    print("Доступные метки:")
    for l in labels:
        print(f"- {l} ({sum(row[-1] == l for row in data)})")

    choice = input("\nВведите имя метки: ").strip()
    items = [(i, row) for i, row in enumerate(data) if row[-1] == choice]

    if not items:
        print("Нет изображений с такой меткой.")
        exit()

    del_idx = []
    for local_idx, (global_idx, row) in enumerate(items, start=1):
        values = [float(x) for x in row[:-1]]
        px = [values[j:j+SIZE] for j in range(0, len(values), SIZE)]
        print(f"\nИзображение {local_idx}, метка: {row[-1]}")
        for y in px:
            print(''.join(block(p) for p in y))
        if input("Enter — дальше, 'delete' — удалить: ").strip().lower() == "delete":
            del_idx.append(global_idx)

    if del_idx:
        data = [r for j, r in enumerate(data) if j not in del_idx]
        with open(FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(data)
        print(f"Удалено {len(del_idx)} примеров.")

import csv
import numpy as np

FILE = 'dataset.csv'
SIZE = 28

def clean(file, thr=0.0):
    with open(file) as f:
        data = list(csv.reader(f))
    data = [r for r in data if not np.all(np.array(r[:-1], float) <= thr)]
    with open(file, 'w', newline='') as f:
        csv.writer(f).writerows(data)

def block(v):
    g = int(v * 23) + 232
    return f"\033[48;5;{g}m   \033[0m"

clean(FILE)

with open(FILE) as f:
    data = list(csv.reader(f))

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
for i, row in items:
    px = np.array(row[:-1], float).reshape(SIZE, SIZE)
    print(f"\nИзображение {i+1}, метка: {row[-1]}")
    for y in px:
        print(''.join(block(p) for p in y))
    if input("Enter — дальше, 'delete' — удалить: ").strip().lower() == "delete":
        del_idx.append(i)

if del_idx:
    data = [r for j, r in enumerate(data) if j not in del_idx]
    with open(FILE, 'w', newline='') as f:
        csv.writer(f).writerows(data)
    print(f"Удалено {len(del_idx)} примеров.")

print("Готово.")

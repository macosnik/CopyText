from PIL import Image, ImageDraw, ImageFont
from model import Net
import os
import numpy as np

def load_and_binarize(path):
    img = Image.open(path).convert("L")
    w, h = img.size
    pixels = list(img.getdata())
    binary = [
        [1 if pixels[y * w + x] / 255 <= 0.4 else 0 for x in range(w)]
        for y in range(h)
    ]
    return binary, img

def find_objects(binary):
    h, w = len(binary), len(binary[0])
    visited = [[False] * w for _ in range(h)]
    objs, dirs = [], [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if binary[y][x] and not visited[y][x]:
                q, visited[y][x], pix = [(y, x)], True, []
                while q:
                    cy, cx = q.pop(0)
                    pix.append((cx, cy))
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary[ny][nx]
                            and not visited[ny][nx]
                        ):
                            visited[ny][nx] = True
                            q.append((ny, nx))
                objs.append(pix)
    return objs

def preprocess_crop(binary, box):
    x0, y0, x1, y1 = box
    crop = [row[x0:x1 + 1] for row in binary[y0:y1 + 1]]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    img = Image.new("L", (w, h))
    img.putdata([v * 255 for row in crop for v in row])
    img = img.resize((28, 28), Image.LANCZOS)
    return [[p / 255 for p in img.getdata()]]

def load_model(path="model.json"):
    return Net.load(path)

def predict_symbol(vec, net, classes):
    p = net.predict_proba(vec)[0]  # вектор вероятностей
    idx = int(np.argmax(p))
    return classes[idx], float(p[idx])

def draw_objects(binary, objs, net, classes, out_path):
    h, w = len(binary), len(binary[0])
    img = Image.new("RGB", (w, h))
    img.putdata([(v * 255, v * 255, v * 255) for row in binary for v in row])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("fonts/Arial.ttf", size=28)

    for obj in objs:
        xs, ys = [p[0] for p in obj], [p[1] for p in obj]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = max(x1 - x0 + 1, y1 - y0 + 1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = int(cx - side / 2), int(cx + side / 2)
        y0, y1 = int(cy - side / 2), int(cy + side / 2)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)

        vec = preprocess_crop(binary, (x0, y0, x1, y1))
        label, prob = predict_symbol(vec, net, classes)
        text = label if prob >= 0.5 else "-"

        draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
        draw.text((x0, y0 - 20), text, fill="red", font=font)

    img.save(out_path)
    print(f"Сохранено: {out_path}")

def recognize_text(binary, objs, net, classes, line_gap=15, space_gap_factor=1.5):
    symbols = []
    for obj in objs:
        xs, ys = [p[0] for p in obj], [p[1] for p in obj]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = max(x1 - x0 + 1, y1 - y0 + 1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = int(cx - side / 2), int(cx + side / 2)
        y0, y1 = int(cy - side / 2), int(cy + side / 2)

        vec = preprocess_crop(binary, (x0, y0, x1, y1))
        label, prob = predict_symbol(vec, net, classes)
        if prob >= 0.5:
            symbols.append((label, x0, y0, y1, x1 - x0 + 1))

    symbols.sort(key=lambda s: (s[2], s[1]))
    lines, current_line = [], []
    last_y = None
    for sym, x, y0, y1, width in symbols:
        cy = (y0 + y1) // 2
        if last_y is None or abs(cy - last_y) <= line_gap:
            current_line.append((x, sym, width))
            last_y = cy if last_y is None else (last_y + cy) // 2
        else:
            current_line.sort(key=lambda s: s[0])
            lines.append(build_line_with_spaces(current_line, space_gap_factor))
            current_line = [(x, sym, width)]
            last_y = cy
    if current_line:
        current_line.sort(key=lambda s: s[0])
        lines.append(build_line_with_spaces(current_line, space_gap_factor))
    return lines

def build_line_with_spaces(symbols, space_gap_factor):
    line = ""
    for i, (x, sym, width) in enumerate(symbols):
        if i > 0:
            prev_x, _, prev_w = symbols[i - 1]
            gap = x - (prev_x + prev_w)
            avg_w = (prev_w + width) / 2
            if gap > avg_w * space_gap_factor:
                line += " "
        line += sym
    return line

if __name__ == "__main__":
    # Загружаем одну модель и список классов
    net = load_model("model.json")
    # Восстанови список классов в том же порядке, что при обучении
    classes = list("0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    binary, _ = load_and_binarize("test.png")
    objs = find_objects(binary)

    lines = recognize_text(binary, objs, net, classes)
    print("Распознанный текст:")
    for line in lines:
        print(line)

    draw_objects(binary, objs, net, classes, "output.png")

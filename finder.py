from PIL import Image, ImageDraw
from model import Net
import os


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
    img = img.resize((20, 20), Image.LANCZOS)
    return [[p / 255 for p in img.getdata()]]


def load_models(path="models"):
    return {
        f[6:-5]: Net.load(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".json")
    }


def draw_objects(binary, objs, models, out_path):
    h, w = len(binary), len(binary[0])
    img = Image.new("RGB", (w, h))
    img.putdata([(v * 255, v * 255, v * 255) for row in binary for v in row])
    draw = ImageDraw.Draw(img)

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

        best_label, best_prob = None, 0
        for lbl, net in models.items():
            p = net.predict_proba(vec)[0]
            prob = p[1] if len(p) == 2 else max(p)
            if prob > best_prob:
                best_prob, best_label = prob, lbl

        text = (
            f"{best_label} ({best_prob:.2f})"
            # f"{best_label}"
            if best_prob >= 0.90
            else f"unknown ({best_prob:.2f})"
        )

        print(text)

        draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
        draw.text((x0, y0 - 12), text, fill="red")

    img.save(out_path)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    models = load_models("models")
    binary, _ = load_and_binarize("test_2.png")
    objs = find_objects(binary)
    draw_objects(binary, objs, models, "output.png")

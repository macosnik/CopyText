import os
import numpy as np
from PIL import Image, ImageDraw
from model import Net


def load_and_binarize(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, float) / 255
    return (arr <= 0.4).astype(np.uint8), img


def find_objects(binary):
    h, w = binary.shape
    visited = np.zeros_like(binary, bool)
    objs, dirs = [], [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                q, visited[y, x], pix = [(y, x)], True, []
                while q:
                    cy, cx = q.pop(0)
                    pix.append((cx, cy))
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            q.append((ny, nx))
                objs.append(pix)
    return objs


def preprocess_crop(binary, box):
    x0, y0, x1, y1 = box
    crop = binary[y0:y1 + 1, x0:x1 + 1]
    img = Image.fromarray(crop * 255).convert("L").resize((20, 20), Image.LANCZOS)
    return (np.array(img, float) / 255).flatten()[None, :]


def load_models(path="models"):
    return {
        f[6:-5]: Net.load(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".json")
    }


def draw_objects(binary, objs, models, out_path):
    img = Image.fromarray((binary * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for obj in objs:
        xs, ys = [p[0] for p in obj], [p[1] for p in obj]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

        side = max(x1 - x0 + 1, y1 - y0 + 1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        x0, x1 = int(cx - side / 2), int(cx + side / 2)
        y0, y1 = int(cy - side / 2), int(cy + side / 2)

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(binary.shape[1] - 1, x1), min(binary.shape[0] - 1, y1)

        vec = preprocess_crop(binary, (x0, y0, x1, y1))

        best_label, best_prob = None, 0
        for lbl, net in models.items():
            p = net.predict_proba(vec)[0]
            prob = p[1] if len(p) == 2 else p.max()
            if prob > best_prob:
                best_prob, best_label = prob, lbl

        text = (
            f"{best_label} ({best_prob:.2f})"
            if best_prob >= 0.9
            else f"unknown ({best_prob:.2f})"
        )

        draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
        draw.text((x0, y0 - 12), text, fill="red")

    img.save(out_path)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    models = load_models("models")
    binary, _ = load_and_binarize("input.png")
    objs = find_objects(binary)
    draw_objects(binary, objs, models, "output.png")

import numpy as np
from PIL import Image, ImageDraw
from model import Net

def load_and_binarize(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    binary = (arr <= 0.4).astype(np.uint8)
    return binary, img

def find_objects(binary):
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    objects = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    for y in range(h):
        for x in range(w):
            if binary[y,x] == 1 and not visited[y,x]:
                queue = [(y,x)]
                visited[y,x] = True
                pixels = []
                while queue:
                    cy,cx = queue.pop(0)
                    pixels.append((cx,cy))
                    for dy,dx in dirs:
                        ny,nx = cy+dy, cx+dx
                        if 0<=ny<h and 0<=nx<w and binary[ny,nx]==1 and not visited[ny,nx]:
                            visited[ny,nx] = True
                            queue.append((ny,nx))
                objects.append(pixels)
    return objects

def preprocess_crop(binary, bbox):
    x_min, y_min, x_max, y_max = bbox
    crop = binary[y_min:y_max+1, x_min:x_max+1]
    img = Image.fromarray((1 - crop) * 255).convert("L")
    img = img.resize((28,28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()[None, :]

def draw_objects(binary, objects, net, orig_img, out_path):
    img = orig_img.convert("RGB")
    draw = ImageDraw.Draw(img)

    for obj in objects:
        xs = [p[0] for p in obj]
        ys = [p[1] for p in obj]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min + 1
        h = y_max - y_min + 1
        side = max(w, h)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        x_min_sq = int(cx - side/2)
        x_max_sq = int(cx + side/2)
        y_min_sq = int(cy - side/2)
        y_max_sq = int(cy + side/2)

        x_min_sq = max(0, x_min_sq)
        y_min_sq = max(0, y_min_sq)
        x_max_sq = min(binary.shape[1]-1, x_max_sq)
        y_max_sq = min(binary.shape[0]-1, y_max_sq)

        vec = preprocess_crop(binary, (x_min_sq,y_min_sq,x_max_sq,y_max_sq))
        probs = net.predict_proba(vec)[0]
        cls = np.argmax(probs)
        prob = probs[cls]

        if prob >= 0.98:
            label = f"{cls} ({prob:.2f})"
        else:
            label = f"unknown ({prob:.2f})"

        draw.rectangle([x_min_sq, y_min_sq, x_max_sq, y_max_sq], outline="red", width=2)
        draw.text((x_min_sq, y_min_sq-12), label, fill="red")

    img.save(out_path)
    print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    net = Net.load("model.json")
    binary, orig_img = load_and_binarize("input.png")
    objects = find_objects(binary)
    draw_objects(binary, objects, net, orig_img, "output.png")

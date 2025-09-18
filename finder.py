import numpy as np
from PIL import Image, ImageDraw
from model import Net

def load_and_binarize(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, np.float32) / 255
    return (arr <= 0.4).astype(np.uint8), img

def find_objects(binary):
    h, w = binary.shape
    visited = np.zeros((h, w), bool)
    objs, dirs = [], [(-1,0),(1,0),(0,-1),(0,1)]
    for y in range(h):
        for x in range(w):
            if binary[y,x] and not visited[y,x]:
                q, pixels = [(y,x)], []
                visited[y,x] = True
                while q:
                    cy,cx = q.pop()
                    pixels.append((cx,cy))
                    for dy,dx in dirs:
                        ny,nx = cy+dy, cx+dx
                        if 0<=ny<h and 0<=nx<w and binary[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            q.append((ny,nx))
                objs.append(pixels)
    return objs

def preprocess_crop(binary, box):
    x0,y0,x1,y1 = box
    crop = binary[y0:y1+1, x0:x1+1] * 255
    img = Image.fromarray(crop).resize((28,28), Image.LANCZOS)
    return (np.array(img, np.float32)/255).flatten()[None,:]

def draw_objects(binary, objs, net, orig, out_path):
    img, draw = orig.convert("RGB"), ImageDraw.Draw(orig.convert("RGB"))

    for obj in objs:
        xs, ys = [p[0] for p in obj], [p[1] for p in obj]
        x0,x1,y0,y1 = min(xs),max(xs),min(ys),max(ys)
        side = max(x1-x0+1, y1-y0+1)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        x0,y0 = max(0,int(cx-side/2)), max(0,int(cy-side/2))
        x1,y1 = min(binary.shape[1]-1,int(cx+side/2)), min(binary.shape[0]-1,int(cy+side/2))

        vec = preprocess_crop(binary,(x0,y0,x1,y1))
        probs = net.predict_proba(vec)[0]
        cls, prob = probs.argmax(), probs.max()
        label = f"{cls} ({prob:.2f})" if prob>=0.98 else f"unknown ({prob:.2f})"

        draw.rectangle([x0,y0,x1,y1], outline="red", width=2)
        draw.text((x0,y0-12), label, fill="red")

    img.save(out_path)
    print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    net = Net.load("model.json")
    binary, orig = load_and_binarize("input.png")
    objs = find_objects(binary)
    draw_objects(binary, objs, net, orig, "output.png")

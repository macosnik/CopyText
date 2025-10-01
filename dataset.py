import csv, os, random
from PIL import Image, ImageDraw, ImageFont

def draw_char(char, font, canvas_size=64):
    pad = canvas_size // 4
    w, h = canvas_size + 2 * pad, canvas_size + 2 * pad
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), char, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (w - tw) // 2 - box[0], (h - th) // 2 - box[1]
    draw.text((x, y), char, fill=255, font=font)
    return img

def crop_to_content(img):
    box = img.getbbox()
    return img.crop(box) if box else img

def fit_to_square(img, size=28):
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("L", (size, size), 0)
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas

def img_to_bits(img):
    return [round(p / 255, 1) for p in img.getdata()]

def build_dataset(fonts, filename, chars):
    rows = []
    for ch in chars:
        for font_path in fonts:
            font = ImageFont.truetype(font_path, 64)
            img = draw_char(ch, font, canvas_size=64)
            img = crop_to_content(img)
            img = fit_to_square(img, 28)
            rows.append(img_to_bits(img) + [ch])
    random.shuffle(rows)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"p{i}" for i in range(28*28)] + ["label"]
        writer.writerow(header)
        writer.writerows(rows)
    print("Готово.")

if __name__ == "__main__":
    FONT_LIBRARY = "fonts_rus"
    NUMS = "1234567890"
    ENG_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ENG_LOWER = "abcdefghijklmnopqrstuvwxyz"
    RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    fonts = [os.path.join(os.path.dirname(__file__), FONT_LIBRARY, f) for f in os.listdir(FONT_LIBRARY) if f.lower().endswith((".ttf", ".otf"))]
    build_dataset(fonts, "dataset.csv", NUMS + RUS_UPPER + RUS_LOWER)

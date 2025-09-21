import csv, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_digit(char, font, size, bold):
    pad = size // 4
    w, h = size + 2 * pad, size + 2 * pad
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), char, font=font, stroke_width=bold)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (w - tw) // 2 - box[0], (h - th) // 2 - box[1]
    draw.text((x, y), char, fill=255, font=font, stroke_width=bold, stroke_fill=255)
    return img

def shear_x(img, deg):
    if deg == 0: return img
    k = math.tan(math.radians(deg))
    w, h = img.size
    shift = abs(k) * h
    return img.transform((int(w + shift), h), Image.AFFINE,
                         (1, k, -shift/2 if k > 0 else shift/2, 0, 1, 0),
                         resample=Image.NEAREST, fillcolor=0)

def shear_y(img, deg):
    if deg == 0: return img
    k = math.tan(math.radians(deg))
    w, h = img.size
    shift = abs(k) * w
    return img.transform((w, int(h + shift)), Image.AFFINE,
                         (1, 0, 0, k, 1, -shift/2 if k > 0 else shift/2),
                         resample=Image.NEAREST, fillcolor=0)

def stretch_x(img, factor):
    if factor == 1.0: return img
    w, h = img.size
    return img.transform((int(w * factor), h), Image.AFFINE,
                         (factor, 0, 0, 0, 1, 0),
                         resample=Image.NEAREST, fillcolor=0)

def stretch_y(img, factor):
    if factor == 1.0: return img
    w, h = img.size
    return img.transform((w, int(h * factor)), Image.AFFINE,
                         (1, 0, 0, 0, factor, 0),
                         resample=Image.NEAREST, fillcolor=0)

def crop(img):
    box = img.getbbox()
    return img.crop(box) if box else img

def to_binary(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.uint8)
    return Image.fromarray(arr)

def fit_to_square(img, size):
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("L", (size, size), 0)
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas

def make_digit(d, font_name, size, bold, sx=0, sy=0, fx=1.0, fy=1.0):
    work = size * 2
    font = ImageFont.truetype(font_name, size=work)
    img = draw_digit(str(d), font, work, bold)
    img = shear_x(img, sx)
    img = shear_y(img, sy)
    img = stretch_x(img, fx)
    img = stretch_y(img, fy)
    img = to_binary(img)
    img = crop(img)
    img = fit_to_square(img, size)
    return img

def build_dataset(fonts, out_csv, size, digits, bolds, slant_x, slant_y, scale_x, scale_y):
    rows = []
    for font in fonts:
        for d in digits:
            for b in range(bolds + 1):
                for sx in slant_x:
                    for sy in slant_y:
                        for fx in scale_x:
                            for fy in scale_y:
                                try:
                                    img = make_digit(d, font, size, b, sx, sy, fx, fy)
                                except:
                                    continue
                                arr = np.array(img, dtype=np.uint8).flatten()
                                rows.append(arr.tolist() + [d])
    random.shuffle(rows)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"p{i}" for i in range(size * size)] + ["label"]
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == "__main__":
    with open("fonts.json") as f:
        fonts = json.load(f)["fonts"]
    build_dataset(fonts, "dataset.csv",
                  size=20,
                  digits="0123456789",
                  bolds=0,
                  slant_x=(-12, -6, 0, 6, 12),
                  slant_y=(-6, 0, 6),
                  scale_x=(1.0, 1.2),
                  scale_y=(1.0, 1.3))

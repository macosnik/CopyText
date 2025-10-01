import csv, random, math, os
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

def to_binary_image(img):
    w, h = img.size
    data = [255 if p > 127 else 0 for p in img.getdata()]
    out = Image.new("L", (w, h), 0)
    out.putdata(data)
    return out

def fit_to_square(img, size):
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

def make_digit(d, font_name, sx, sy, fx, fy):
    font = ImageFont.truetype(font_name, 64)
    img = draw_digit(str(d), font, 64, 0)
    img = shear_x(img, sx)
    img = shear_y(img, sy)
    img = stretch_x(img, fx)
    img = stretch_y(img, fy)
    img = to_binary_image(img)
    img = crop(img)
    img = fit_to_square(img, 20)
    return img

def progress_bar(current, total, prefix=""):
    filled = int(100 * current // total)
    bar = "█" * filled + " " * (100 - filled)
    percent = (current / total) * 100
    print(f"\r\033[31m{prefix} |{bar}| {percent:6.2f}%\033[0m", end="")

def img_to_bits(img):
    return [1 if p > 127 else 0 for p in img.getdata()]

def build_dataset(fonts, filename, digits, slant_x=None, slant_y=None, scale_x=None, scale_y=None):
    slant_x, slant_y = [0] if slant_x is None else slant_x, [0] if slant_y is None else slant_y
    scale_x, scale_y = [1.0] if scale_x is None else scale_x, [1.0] if scale_y is None else scale_y
    rows = []
    per_symbol = len(fonts) * len(slant_x) * len(slant_y) * len(scale_x) * len(scale_y)
    for d in digits:
        done = 0
        for font in fonts:
            for sx in slant_x:
                for sy in slant_y:
                    for fx in scale_x:
                        for fy in scale_y:
                            done += 1
                            img = make_digit(d, font, sx, sy, fx, fy)
                            rows.append(img_to_bits(img) + [d])
                            progress_bar(done, per_symbol, prefix=f"Символ {d}")
    random.shuffle(rows)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"p{i}" for i in range(400)] + ["label"]
        writer.writerow(header)
        writer.writerows(rows)
    print("\rГотово.")

if __name__ == "__main__":
    NUMS = "1234567890"

    ENG_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ENG_LOWER = "abcdefghijklmnopqrstuvwxyz"

    RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

    ALPHABET = NUMS + ENG_UPPER + ENG_LOWER + RUS_UPPER + RUS_LOWER

    fonts = [os.path.join(os.path.dirname(__file__), "fonts", font) for font in os.listdir("fonts") if not font.startswith(".")]

    build_dataset(fonts, "dataset.csv",
                  NUMS + RUS_UPPER + RUS_LOWER,
                  None, None, None, None)

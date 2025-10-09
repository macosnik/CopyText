import os, csv, random
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed

def draw_word(word, font_path, size=(128,32)):
    img = Image.new("L", size, 0)
    font_size = size[1]
    while font_size > 8:
        font = ImageFont.truetype(font_path, font_size)
        w,h = font.getbbox(word)[2:]
        if w <= size[0] and h <= size[1]:
            break
        font_size -= 1
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), word, font=font)
    x,y = (size[0] - (box[2] - box[0])) // 2 - box[0], (size[1] - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), word, fill=255, font=font)
    return img

def process_one(i, word, font_path, size):
    img = draw_word(word, font_path, size)
    path = f"words/{i:06d}.png"
    img.save(path)
    return [path, word, len(word)]

def build_dataset(fonts, filename, words, n_samples, size, workers):
    os.makedirs("words", exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i in range(n_samples):
            word = random.choice(words)
            font_path = random.choice(fonts)
            futures.append(ex.submit(process_one, i, word, font_path, size))
        total = len(futures)
        for n, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if n % 10 == 0 or n == total:
                print(f"\r{n}/{total}", end="")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_length"])
        w.writerows(rows)

if __name__ == "__main__":
    fonts = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.lower().endswith((".ttf", ".otf"))]
    NUMS = "1234567890"
    RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    SYMBOLS = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    ALPHABET = NUMS + RUS_UPPER + RUS_LOWER + SYMBOLS
    def random_word(min_len=3, max_len=12):
        return "".join(random.choice(ALPHABET) for _ in range(random.randint(min_len, max_len)))
    words=[random_word() for _ in range(100000)]
    build_dataset(fonts, "dataset.csv", words, 100000, (128,32), os.cpu_count())

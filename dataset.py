import os, random
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from concurrent.futures import ProcessPoolExecutor, as_completed

def draw_word(word, font_path, size=(128,32)):
    img = Image.new("L", size, 0)
    target_w, target_h = size

    font_size = target_h
    while font_size > 1:
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(word)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target_w and h <= target_h:
            break
        font_size -= 1

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (target_w - text_w) // 2 - bbox[0]
    y = (target_h - text_h) // 2 - bbox[1]
    draw.text((x, y), word, fill=255, font=font)

    return img

def process_one(i, word, font_path, size):
    img = draw_word(word, font_path, size)
    path = f"dataset/{i:06d}.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("label", word)
    img.save(path, pnginfo=meta)
    return path

def random_fraze(word, alphabet, symbols, nums, prob_random=0.15, prob_numeric=0.15):
    if random.random() < prob_random:
        length = random.randint(3, 12)
        return "".join(random.choice(alphabet) for _ in range(length))

    if random.random() < prob_numeric:
        length = random.randint(2, 8)
        return "".join(random.choice(nums) for _ in range(length))

    if random.random() < 0.3:
        word = random.choice(symbols) + word
    if random.random() < 0.3:
        word = word + random.choice(symbols)

    if random.random() < 0.3:
        pos = random.randint(0, len(word))
        word = word[:pos] + random.choice(nums) + word[pos:]

    return word

def build_dataset(fonts, words, n_samples, size, workers, alphabet, symbols, nums):
    os.makedirs("dataset", exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i in range(n_samples):
            base_word = random.choice(words)
            word = random_fraze(base_word, alphabet, symbols, nums)
            font_path = random.choice(fonts)
            futures.append(ex.submit(process_one, i, word, font_path, size))
        total = len(futures)
        for n, f in enumerate(as_completed(futures), 1):
            f.result()
            if n % 100 == 0 or n == total:
                print(f"\r{n}/{total}", end="")

if __name__ == "__main__":
    fonts = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.lower().endswith((".ttf", ".otf"))]
    NUMS = "1234567890"
    RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    SYMBOLS = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    ALPHABET = NUMS + RUS_UPPER + RUS_LOWER + SYMBOLS

    with open("russian.txt", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    build_dataset(fonts, words, 10000, (128,32), os.cpu_count(), ALPHABET, SYMBOLS, NUMS)

from PIL import Image
import pygame, csv, os

pygame.init()
screen = pygame.display.set_mode((500, 530))
pygame.display.set_caption("Рисовалка")

symbols = [
    '0','1-0','1-1','1-2','2-0','2-1','3','4-0','4-1','5','6','7-0','7-1','8','9',
    'а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я'
]

groups = {}
for s in symbols:
    base = s.split('-')[0] if '-' in s else s
    groups.setdefault(base, []).append(s)

current_symbol, current_base, current_variant_idx = None, None, 0
radius = 20

canvas = pygame.Surface((500, 500))
canvas.fill((0, 0, 0))

font = pygame.font.SysFont("Arial", 20)
dataset_file = "dataset.csv"

if not os.path.exists(dataset_file):
    headers = [f"p{i}" for i in range(400)] + ["label"]
    with open(dataset_file, "w", newline="") as f:
        csv.writer(f).writerow(headers)

label_counts = {s: 0 for s in symbols}
with open(dataset_file) as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if row and row[-1] in label_counts:
            label_counts[row[-1]] += 1


def draw_ui():
    screen.fill((50, 50, 50))
    screen.blit(canvas, (0, 30))
    count = label_counts.get(current_symbol, 0) if current_symbol else 0
    text = f"Символ: {current_symbol or 'не выбран'} | Толщина: {radius} | Меток: {count}"
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, (10, 5))


def save_image():
    if not current_symbol:
        return

    raw = pygame.image.tostring(canvas, "RGB")
    img = Image.frombytes("RGB", (500, 500), raw)
    gray = img.convert("L")

    bbox = gray.getbbox()
    if not bbox:
        return

    cropped = gray.crop(bbox)
    side = max(cropped.size)
    square = Image.new("L", (side, side), 0)
    x_off = (side - cropped.size[0]) // 2
    y_off = (side - cropped.size[1]) // 2
    square.paste(cropped, (x_off, y_off))

    img20 = square.resize((20, 20), Image.LANCZOS)
    pixels = list(img20.getdata())

    row = [f"{round(p/255*10)/10:.1f}" for p in pixels] + [current_symbol]
    with open(dataset_file, "a", newline="") as f:
        csv.writer(f).writerow(row)

    label_counts[current_symbol] += 1
    canvas.fill((0, 0, 0))


running, drawing = True, False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                canvas.fill((0, 0, 0))
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                save_image()
            else:
                ch = event.unicode
                if ch in groups:
                    if current_base == ch:
                        current_variant_idx = (current_variant_idx + 1) % len(groups[ch])
                    else:
                        current_base, current_variant_idx = ch, 0
                    current_symbol = groups[ch][current_variant_idx]
            if event.key == pygame.K_UP:
                radius += 1
            elif event.key == pygame.K_DOWN:
                radius = max(1, radius - 1)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False

    if drawing:
        x, y = pygame.mouse.get_pos()
        if y >= 30:
            pygame.draw.circle(canvas, (255, 255, 255), (x, y-30), radius)

    draw_ui()
    pygame.display.flip()

pygame.quit()

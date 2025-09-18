from PIL import Image
import pygame
import numpy as np
import csv

pygame.init()

WIDTH, HEIGHT = 500, 530
CANVAS_SIZE = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Рисовалка")

symbols = list("0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
current_symbol = None
radius = 20

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill((0, 0, 0))

font = pygame.font.SysFont("Arial", 20)

def draw_ui():
    screen.fill((50, 50, 50))
    screen.blit(canvas, (0, 30))
    text = f"Символ: {current_symbol or 'не выбран'} | Толщина: {radius}"
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, (10, 5))

def save_image():
    if not current_symbol:
        return
    arr = np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    coords = np.argwhere(gray > 0)
    if coords.size == 0:
        return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = gray[y0:y1, x0:x1]
    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cropped
    img = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    arr = (np.array(img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    with open("dataset.csv", "a", newline="") as f:
        csv.writer(f).writerow(arr.flatten().tolist() + [current_symbol])
    canvas.fill((0, 0, 0))

running, drawing = True, False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                canvas.fill((0, 0, 0))
            elif event.key == pygame.K_SPACE:
                save_image()
            elif event.unicode in symbols:
                current_symbol = event.unicode
            elif event.key == pygame.K_UP:
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

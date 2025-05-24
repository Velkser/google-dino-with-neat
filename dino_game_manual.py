import pygame
import random
import sys

# Инициализация Pygame
pygame.init()

# Параметры экрана
WIDTH, HEIGHT = 800, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Google Dino Game")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Переменные динозавра
dino_width, dino_height = 40, 40
dino_x, dino_y = 50, HEIGHT - dino_height - 20
dino_jump = False
jump_height = 12
velocity_y = 0
gravity = 1

# Кактус
cactus_width, cactus_height = 20, 40
cactus_x = WIDTH
cactus_y = HEIGHT - cactus_height - 20
cactus_speed = 12

# Счёт
score = 0
font = pygame.font.SysFont(None, 36)

clock = pygame.time.Clock()

def draw_window(dino_y, cactus_x, score):
    win.fill(WHITE)
    pygame.draw.rect(win, BLACK, (dino_x, dino_y, dino_width, dino_height))  # динозавр
    pygame.draw.rect(win, (34, 139, 34), (cactus_x, cactus_y, cactus_width, cactus_height))  # кактус
    pygame.draw.line(win, BLACK, (0, HEIGHT - 20), (WIDTH, HEIGHT - 20), 2)  # земля

    score_text = font.render(f"Score: {score}", True, BLACK)
    win.blit(score_text, (10, 10))
    pygame.display.update()

def main():
    global dino_y, dino_jump, velocity_y, cactus_x, score

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and not dino_jump:
                if event.key == pygame.K_SPACE:
                    dino_jump = True
                    velocity_y = -jump_height

        # Прыжок
        if dino_jump:
            dino_y += velocity_y
            velocity_y += gravity
            if dino_y >= HEIGHT - dino_height - 20:
                dino_y = HEIGHT - dino_height - 20
                dino_jump = False

        # Движение кактуса
        cactus_x -= cactus_speed
        if cactus_x < -cactus_width:
            cactus_x = WIDTH + random.randint(100, 300)
            score += 1

        # Проверка столкновений
        dino_rect = pygame.Rect(dino_x, dino_y, dino_width, dino_height)
        cactus_rect = pygame.Rect(cactus_x, cactus_y, cactus_width, cactus_height)
        if dino_rect.colliderect(cactus_rect):
            print("Game Over! Final Score:", score)
            pygame.quit()
            sys.exit()

        draw_window(dino_y, cactus_x, score)

main()

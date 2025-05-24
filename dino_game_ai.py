import pygame
import random
import sys
import math


class NeuralNetwork:
    def __init__(self):
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.bias = random.uniform(-1, 1)

    def predict(self, inputs):
        x = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return sigmoid(x)

    def mutate(self):
        for i in range(len(self.weights)):
            if random.random() < MUTATION_RATE:
                self.weights[i] += random.uniform(-0.5, 0.5)
        if random.random() < MUTATION_RATE:
            self.bias += random.uniform(-0.5, 0.5)

    def copy(self):
        new_net = NeuralNetwork()
        new_net.weights = self.weights[:]
        new_net.bias = self.bias
        return new_net

class DinoAI:
    def __init__(self):
        self.y = ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.alive = True
        self.brain = NeuralNetwork()
        self.fitness = 0
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def update(self, cactus):
        if not self.alive:
            return

        if cactus:
            distance = (cactus['x'] - dino_x) / WIDTH
            speed_norm = cactus_speed / MAX_SPEED
            jump_state = 1.0 if self.is_jumping else 0.0
            inputs = [distance, speed_norm, jump_state]

            output = self.brain.predict(inputs)
            if output > 0.5 and not self.is_jumping:
                self.is_jumping = True
                self.vel_y = -jump_height

        if self.is_jumping:
            self.y += self.vel_y
            self.vel_y += gravity
            if self.y >= ground_y:
                self.y = ground_y
                self.is_jumping = False

        self.fitness += 1

    def draw(self, surface):
        if self.alive:
            pygame.draw.rect(surface, self.color, (dino_x, self.y, dino_width, dino_height))

class Population:
    def __init__(self):
        self.generation = 1
        self.dinos = [DinoAI() for _ in range(POPULATION_SIZE)]

    def all_dead(self):
        return all(not d.alive for d in self.dinos)

    def evolve(self):
        self.generation += 1
        self.dinos.sort(key=lambda d: d.fitness, reverse=True)
        survivors = self.dinos[:POPULATION_SIZE // 5]  # 20% лучших
        new_dinos = []
        for _ in range(POPULATION_SIZE):
            parent = random.choice(survivors)
            child = DinoAI()
            child.brain = parent.brain.copy()
            child.brain.mutate()
            new_dinos.append(child)
        self.dinos = new_dinos

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Инициализация Pygame
pygame.init()

# Параметры экрана
WIDTH, HEIGHT = 800, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Google Dino AI Evolution")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)

# Настройки
POPULATION_SIZE = 100
MUTATION_RATE = 0.1

# Параметры динозавра
dino_width, dino_height = 20, 20
dino_x = 50
ground_y = HEIGHT - dino_height - 20
jump_height = 10
gravity = 1

# Параметры кактуса
cactus_speed = 5
MAX_SPEED = 10

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)





def spawn_cactus():
    return {
        'x': WIDTH + random.randint(0, 200),
        'width': random.randint(15, 35),
        'height': random.randint(25, 50),
        'passed': False
    }

def draw(cacti, obstacles_passed, max_obstacles=0):
    win.fill(WHITE)
    for cactus in cacti:
        pygame.draw.rect(win, GREEN, (cactus['x'], HEIGHT - cactus['height'] - 20, cactus['width'], cactus['height']))
    pygame.draw.line(win, BLACK, (0, HEIGHT - 20), (WIDTH, HEIGHT - 20), 2)

    for dino in population.dinos:
        dino.draw(win)

    text = font.render(f"Gen: {population.generation} | Alive: {sum(d.alive for d in population.dinos)} | Obstacles Passed: {obstacles_passed} | Max Obstacles Passed: {max_obstacles}", True, BLACK)
    win.blit(text, (10, 10))
    pygame.display.update()

population = Population()
cacti = [spawn_cactus()]
cactus_timer = random.randint(60, 120)
obstacles_passed = 0

def main():

    max_obstacles_passed = 0
    global cactus_timer, cacti, obstacles_passed
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        cactus_timer -= 1
        if cactus_timer <= 0:
            cacti.append(spawn_cactus())
            cactus_timer = random.randint(60, 120)

        for cactus in cacti:
            cactus['x'] -= cactus_speed
            if not cactus['passed'] and cactus['x'] + cactus['width'] < dino_x:
                cactus['passed'] = True
                obstacles_passed += 1

        cacti = [c for c in cacti if c['x'] + c['width'] > 0]

        nearest = None
        for cactus in cacti:
            if cactus['x'] + cactus['width'] >= dino_x:
                nearest = cactus
                break

        for dino in population.dinos:
            if dino.alive:
                dino.update(nearest)
                for cactus in cacti:
                    dino_rect = pygame.Rect(dino_x, dino.y, dino_width, dino_height)
                    cactus_rect = pygame.Rect(cactus['x'], HEIGHT - cactus['height'] - 20, cactus['width'], cactus['height'])
                    if dino_rect.colliderect(cactus_rect):
                        dino.alive = False

        if population.all_dead():
            population.evolve()
            cacti = [spawn_cactus()]
            cactus_timer = random.randint(60, 120)
            obstacles_passed = 0

        if obstacles_passed > max_obstacles_passed:
            max_obstacles_passed = obstacles_passed
        draw(cacti, obstacles_passed, max_obstacles=max_obstacles_passed)

main()

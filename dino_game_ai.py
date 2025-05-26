import pygame
import random
import sys
import math

class NeuralNetwork:
    def __init__(self):
        """
        Initializes the neural network with random weights and a bias.
        The network takes 3 input values.
        """
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.bias = random.uniform(-1, 1)

    def predict(self, inputs):
        """
        Performs a forward pass using the sigmoid activation function.
        
        Args:
            inputs (list): List of 3 input features.
        
        Returns:
            float: Output of the neural network between 0 and 1.
        """
        x = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return sigmoid(x)

    def mutate(self):
        """
        Mutates the weights and bias of the neural network 
        by a small random amount based on the mutation rate.
        """
        for i in range(len(self.weights)):
            if random.random() < MUTATION_RATE:
                self.weights[i] += random.uniform(-0.5, 0.5)
        if random.random() < MUTATION_RATE:
            self.bias += random.uniform(-0.5, 0.5)

    def copy(self):
        """
        Creates a deep copy of the neural network.
        
        Returns:
            NeuralNetwork: A copy of the current neural network.
        """
        new_net = NeuralNetwork()
        new_net.weights = self.weights[:]
        new_net.bias = self.bias
        return new_net

class DinoAI:
    def __init__(self):
        """
        Initializes a DinoAI agent with a neural network brain, 
        initial state, and random color for visualization.
        """
        self.y = ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.alive = True
        self.brain = NeuralNetwork()
        self.fitness = 0
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def update(self, cactus):
        """
        Updates the state of the DinoAI. If alive and a cactus is near, 
        makes a decision whether to jump. Updates physics and fitness.
        
        Args:
            cactus (dict): Dictionary containing the nearest cactus data.
        """
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
        """
        Draws the DinoAI on the given surface if alive.
        
        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        if self.alive:
            pygame.draw.rect(surface, self.color, (dino_x, self.y, dino_width, dino_height))

class Population:
    def __init__(self):
        """
        Initializes a population of DinoAI agents.
        """
        self.generation = 1
        self.dinos = [DinoAI() for _ in range(POPULATION_SIZE)]

    def all_dead(self):
        """
        Checks if all dinos in the population are dead.
        
        Returns:
            bool: True if all dinos are dead, False otherwise.
        """
        return all(not d.alive for d in self.dinos)

    def evolve(self):
        """
        Evolves the current generation using a simplified 
        artificial immune algorithm with cloning, hypermutation,
        and introduction of new random agents.
        """
        self.generation += 1
        self.dinos.sort(key=lambda d: d.fitness, reverse=True)

        clones = []
        total_fitness = sum(d.fitness for d in self.dinos[:POPULATION_SIZE // 2]) + 1e-6

        for dino in self.dinos[:POPULATION_SIZE // 2]:
            n_clones = max(1, int((dino.fitness / total_fitness) * POPULATION_SIZE))
            for _ in range(n_clones):
                clone = DinoAI()
                clone.brain = dino.brain.copy()

                mutation_strength = 1.0 - (dino.fitness / (self.dinos[0].fitness + 1e-6))
                for i in range(len(clone.brain.weights)):
                    if random.random() < MUTATION_RATE:
                        clone.brain.weights[i] += random.uniform(-1, 1) * mutation_strength
                if random.random() < MUTATION_RATE:
                    clone.brain.bias += random.uniform(-1, 1) * mutation_strength

                clones.append(clone)

        num_random = POPULATION_SIZE // 10
        random_dinos = [DinoAI() for _ in range(num_random)]

        clones.sort(key=lambda d: d.fitness, reverse=True)
        next_gen = clones[:POPULATION_SIZE - num_random] + random_dinos

        while len(next_gen) < POPULATION_SIZE:
            next_gen.append(DinoAI())

        self.dinos = next_gen

def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x (float): Input value.
    
    Returns:
        float: Transformed value between 0 and 1.
    """
    return 1 / (1 + math.exp(-x))

def spawn_cactus():
    """
    Spawns a new cactus with random dimensions and position.
    
    Returns:
        dict: A dictionary containing cactus attributes.
    """
    return {
        'x': WIDTH + random.randint(0, 200),
        'width': random.randint(15, 35),
        'height': random.randint(25, 50),
        'passed': False
    }

def draw(cacti, obstacles_passed, max_obstacles=0):
    """
    Renders the game screen including dinos, cacti, and UI stats.
    
    Args:
        cacti (list): List of cactus dictionaries.
        obstacles_passed (int): Number of obstacles passed.
        max_obstacles (int): Highest number of obstacles passed so far.
    """
    win.fill(WHITE)
    for cactus in cacti:
        pygame.draw.rect(win, GREEN, (cactus['x'], HEIGHT - cactus['height'] - 20, cactus['width'], cactus['height']))
    pygame.draw.line(win, BLACK, (0, HEIGHT - 20), (WIDTH, HEIGHT - 20), 2)

    for dino in population.dinos:
        dino.draw(win)

    text = font.render(f"Gen: {population.generation} | Alive: {sum(d.alive for d in population.dinos)} | Obstacles Passed: {obstacles_passed} | Max Obstacles Passed: {max_obstacles}", True, BLACK)
    win.blit(text, (10, 10))
    pygame.display.update()

def main():
    """
    Main game loop. Handles updates, drawing, cactus spawning,
    collision detection, and triggers evolution when all dinos die.
    """
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


pygame.init()

WIDTH, HEIGHT = 800, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Google Dino AI Evolution")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)

POPULATION_SIZE = 100
MUTATION_RATE = 0.1

dino_width, dino_height = 20, 20
dino_x = 50
ground_y = HEIGHT - dino_height - 20
jump_height = 10
gravity = 1

cactus_speed = 5
MAX_SPEED = 10

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


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


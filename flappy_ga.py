"""
Flappy Bird AI with Genetic Algorithm + Tiny Neural Network
Single-file implementation using Python, pygame and numpy.

Features
- Simple Flappy Bird clone environment
- Population of birds each controlled by a small neural net
- Fitness = survival time + pipes passed
- Tournament selection + single-point crossover + gaussian mutation
- On-screen HUD: generation, alive, best score, fps, mutation rate
- Optional fast mode (reduced rendering) and model saving for best genome

Controls
- SPACE: toggle drawing (fast mode on/off)
- R: restart training from generation 0
- S: save current best genome to ./best_genome.npz
- L: load ./best_genome.npz and play a showcase run
- ESC or Q: quit

Dependencies
pip install pygame numpy

Run
python flappy_ga.py

Tested with Python 3.10+
"""
from __future__ import annotations
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

# --------------------- Game Config ---------------------
WIDTH, HEIGHT = 480, 720
GROUND_Y = HEIGHT - 120
FPS = 60
GRAVITY = 0.5
FLAP_VELOCITY = -8.5
PIPE_SPEED = 3.0
PIPE_GAP = 170
PIPE_DIST = 220  # distance between pipe columns
BIRD_X = 100

# --------------------- GA Config ---------------------
INPUT_SIZE = 4  # [dx_to_pipe, dy_to_gap_center, dy_to_top, vy]
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1  # jump probability
POP_SIZE = 80
ELITE_FRAC = 0.1
TOURNAMENT_K = 5
MUTATION_RATE = 0.12
MUTATION_STD = 0.6
BIAS_MUTATION_STD = 0.2
CROSSOVER_RATE = 0.75
MAX_GENERATIONS = 10_000
SEED = 42
SAVE_PATH = "best_genome.npz"

random.seed(SEED)
np.random.seed(SEED)

# --------------------- Helper: Tiny NN ---------------------
class TinyNN:
    """A minimal 2-layer MLP: input->hidden(tanh)->output(sigmoid)."""
    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    @staticmethod
    def init_random(input_size: int, hidden_size: int, output_size: int) -> "TinyNN":
        # Xavier/Glorot-ish init
        w1 = np.random.randn(hidden_size, input_size) * math.sqrt(2 / (input_size + hidden_size))
        b1 = np.zeros((hidden_size,))
        w2 = np.random.randn(output_size, hidden_size) * math.sqrt(2 / (hidden_size + output_size))
        b2 = np.zeros((output_size,))
        return TinyNN(w1, b1, w2, b2)

    def forward(self, x: np.ndarray) -> float:
        h = np.tanh(self.w1 @ x + self.b1)
        y = self.w2 @ h + self.b2
        # sigmoid
        y = 1 / (1 + np.exp(-y))
        return float(y[0])

    # ---- Genome ops ----
    def copy(self) -> "TinyNN":
        return TinyNN(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    @staticmethod
    def crossover(a: "TinyNN", b: "TinyNN") -> "TinyNN":
        def cx(arr1, arr2):
            flat1, flat2 = arr1.reshape(-1), arr2.reshape(-1)
            if len(flat1) != len(flat2):
                raise ValueError("shape mismatch")
            if random.random() < CROSSOVER_RATE:
                p = random.randrange(len(flat1))
                child = np.concatenate([flat1[:p], flat2[p:]])
            else:
                child = flat1.copy()
            return child.reshape(arr1.shape)
        w1 = cx(a.w1, b.w1)
        b1 = cx(a.b1, b.b1)
        w2 = cx(a.w2, b.w2)
        b2 = cx(a.b2, b.b2)
        return TinyNN(w1, b1, w2, b2)

    def mutate(self):
        def m(arr, std):
            mask = np.random.rand(*arr.shape) < MUTATION_RATE
            noise = np.random.randn(*arr.shape) * std
            arr[mask] += noise[mask]
        m(self.w1, MUTATION_STD)
        m(self.w2, MUTATION_STD)
        m(self.b1, BIAS_MUTATION_STD)
        m(self.b2, BIAS_MUTATION_STD)

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @staticmethod
    def load(path: str) -> "TinyNN":
        d = np.load(path)
        return TinyNN(d['w1'], d['b1'], d['w2'], d['b2'])

# --------------------- Game Objects ---------------------
class Bird:
    def __init__(self, brain: TinyNN):
        self.x = BIRD_X
        self.y = HEIGHT * 0.4
        self.vy = 0.0
        self.alive = True
        self.score = 0
        self.time_alive = 0
        self.brain = brain

    def update(self, env: "Game", action: bool):
        if not self.alive:
            return
        if action:
            self.vy = FLAP_VELOCITY
        self.vy += GRAVITY
        self.y += self.vy
        self.time_alive += 1
        # ground / ceiling
        if self.y < 0 or self.y + 24 > GROUND_Y:
            self.alive = False

    def decide(self, obs: np.ndarray) -> bool:
        p = self.brain.forward(obs)
        return p > 0.5

class Pipe:
    def __init__(self, x: float):
        self.x = x
        self.gap_y = random.randint(140, GROUND_Y - 160)
        self.passed = False

    def rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        top_rect = pygame.Rect(int(self.x), 0, 60, int(self.gap_y - PIPE_GAP/2))
        bot_rect = pygame.Rect(int(self.x), int(self.gap_y + PIPE_GAP/2), 60, int(GROUND_Y - (self.gap_y + PIPE_GAP/2)))
        return top_rect, bot_rect

    def update(self):
        self.x -= PIPE_SPEED

# --------------------- Environment ---------------------
class Game:
    def __init__(self, draw: bool = True):
        self.draw = draw
        self.reset()
        if draw:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy Bird GA AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 20)
        else:
            self.screen = None
            self.clock = None
            self.font = None

    def reset(self):
        self.pipes: List[Pipe] = []
        self.spawn_pipe(400)
        self.spawn_pipe(400 + PIPE_DIST)
        self.spawn_pipe(400 + 2*PIPE_DIST)
        self.scroll = 0.0
        self.frame = 0
        self.best_score = 0

    def spawn_pipe(self, x):
        self.pipes.append(Pipe(x))

    def update_pipes(self):
        if not self.pipes:
            self.spawn_pipe(WIDTH + 100)
        # move
        for p in self.pipes:
            p.update()
        # recycle
        if self.pipes and self.pipes[0].x < -70:
            self.pipes.pop(0)
        # ensure constant spacing
        while self.pipes and self.pipes[-1].x < WIDTH + PIPE_DIST:
            self.spawn_pipe(self.pipes[-1].x + PIPE_DIST)

    def get_next_pipe(self, x: float) -> Pipe:
        for p in self.pipes:
            if p.x + 60 >= x:
                return p
        return self.pipes[0]

    def get_obs(self, bird: Bird) -> np.ndarray:
        p = self.get_next_pipe(bird.x)
        dx = (p.x + 30) - bird.x
        dy_gap = (p.gap_y) - bird.y
        dy_top = (p.gap_y - PIPE_GAP/2) - bird.y
        vx = bird.vy
        # normalize roughly
        x = np.array([
            dx / WIDTH,
            dy_gap / HEIGHT,
            dy_top / HEIGHT,
            vx / 10.0,
        ], dtype=np.float32)
        return x

    def step(self, birds: List[Bird]) -> None:
        self.frame += 1
        self.update_pipes()
        # collisions + scoring
        for b in birds:
            if not b.alive:
                continue
            p = self.get_next_pipe(b.x)
            top_rect, bot_rect = p.rects()
            bird_rect = pygame.Rect(int(b.x-17), int(b.y-12), 34, 24)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                b.alive = False
            # passed pipe center
            if not p.passed and (p.x + 30) < b.x:
                p.passed = True
                b.score += 1
                self.best_score = max(self.best_score, b.score)

    def draw_frame(self, birds: List[Bird], gen: int, alive: int, mutation_rate: float, fps_now: float):
        if not self.draw:
            return
        self.screen.fill((135, 206, 235))  # sky
        # ground
        pygame.draw.rect(self.screen, (222, 184, 135), pygame.Rect(0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        # pipes
        for p in self.pipes:
            top_rect, bot_rect = p.rects()
            pygame.draw.rect(self.screen, (0, 200, 0), top_rect)
            pygame.draw.rect(self.screen, (0, 200, 0), bot_rect)
        # birds
        for b in birds:
            if b.alive:
                pygame.draw.ellipse(self.screen, (255, 255, 0), pygame.Rect(int(b.x-17), int(b.y-12), 34, 24))
        # HUD
        lines = [
            f"Gen: {gen}",
            f"Alive: {alive}/{len(birds)}",
            f"Best Score: {self.best_score}",
            f"MutRate: {mutation_rate:.2f}",
            f"FPS: {fps_now:.1f}",
            f"Controls: SPACE=fast, R=reset, S=save, L=load, Q/Esc=quit",
        ]
        for i, txt in enumerate(lines):
            s = self.font.render(txt, True, (0, 0, 0))
            self.screen.blit(s, (10, 10 + 22*i))
        pygame.display.flip()

# --------------------- GA Utilities ---------------------
@dataclass
class Individual:
    brain: TinyNN
    fitness: float = 0.0


def evaluate_population(env: Game, pop: List[Individual], draw: bool, max_steps: int = 10_000) -> None:
    birds = [Bird(ind.brain) for ind in pop]
    steps = 0
    last_time = time.time()
    while steps < max_steps and any(b.alive for b in birds):
        # events
        for event in pygame.event.get() if env.draw else []:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
        keys = pygame.key.get_pressed() if env.draw else None
        if env.draw and (keys[pygame.K_q] or keys[pygame.K_ESCAPE]):
            pygame.quit(); sys.exit(0)
        # toggle draw
        if env.draw and keys[pygame.K_SPACE]:
            env.draw = False
        # per-bird decision/update
        for b in birds:
            if not b.alive:
                continue
            obs = env.get_obs(b)
            action = b.decide(obs)
            b.update(env, action)
        env.step(birds)
        steps += 1
        # rendering
        if env.draw:
            now = time.time(); dt = now - last_time; last_time = now
            fps_now = 1.0/dt if dt > 0 else FPS
            env.draw_frame(birds, gen=current_generation[0], alive=sum(b.alive for b in birds), mutation_rate=MUTATION_RATE, fps_now=fps_now)
            env.clock.tick(FPS)
    # assign fitness
    for ind, b in zip(pop, birds):
        ind.fitness = b.time_alive + b.score * 500  # reward pipes passed heavily


def tournament_selection(pop: List[Individual], k: int) -> Individual:
    cand = random.sample(pop, k)
    return max(cand, key=lambda i: i.fitness)


def make_next_generation(pop: List[Individual]) -> List[Individual]:
    pop = sorted(pop, key=lambda i: i.fitness, reverse=True)
    n = len(pop)
    elites = pop[: max(1, int(ELITE_FRAC * n))]
    next_pop: List[Individual] = [Individual(e.brain.copy()) for e in elites]

    while len(next_pop) < n:
        p1 = tournament_selection(pop, TOURNAMENT_K)
        p2 = tournament_selection(pop, TOURNAMENT_K)
        child_brain = TinyNN.crossover(p1.brain, p2.brain)
        child_brain.mutate()
        next_pop.append(Individual(child_brain))
    return next_pop[:n]

# Track generation index globally for HUD
current_generation = [0]

# --------------------- Showcase Run ---------------------
def showcase(env: Game, brain: TinyNN):
    bird = Bird(brain)
    running = True
    last_time = time.time()
    while running and bird.alive:
        for event in pygame.event.get() if env.draw else []:
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
        keys = pygame.key.get_pressed() if env.draw else None
        if env.draw and (keys[pygame.K_q] or keys[pygame.K_ESCAPE]):
            pygame.quit(); sys.exit(0)
        obs = env.get_obs(bird)
        bird.update(env, bird.decide(obs))
        env.step([bird])
        if env.draw:
            now = time.time(); dt = now - last_time; last_time = now
            fps_now = 1.0/dt if dt > 0 else FPS
            env.draw_frame([bird], gen=-1, alive=int(bird.alive), mutation_rate=MUTATION_RATE, fps_now=fps_now)
            env.clock.tick(FPS)

# --------------------- Main Loop ---------------------
def main():
    draw = True
    env = Game(draw=draw)

    # init population
    pop: List[Individual] = [
        Individual(TinyNN.init_random(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE))
        for _ in range(POP_SIZE)
    ]

    best_fitness = -1e9
    best_brain = pop[0].brain.copy()

    while current_generation[0] < MAX_GENERATIONS:
        # Input handling for global hotkeys
        if env.draw:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        env.draw = False
                    elif event.key == pygame.K_r:
                        # restart from scratch
                        pop = [Individual(TinyNN.init_random(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)) for _ in range(POP_SIZE)]
                        current_generation[0] = 0
                        best_fitness = -1e9
                        best_brain = pop[0].brain.copy()
                        env.reset()
                    elif event.key == pygame.K_s:
                        best_brain.save(SAVE_PATH)
                        print(f"Saved best genome to {SAVE_PATH}")
                    elif event.key == pygame.K_l:
                        if os.path.exists(SAVE_PATH):
                            try:
                                best_brain = TinyNN.load(SAVE_PATH)
                                print("Loaded best genome; running showcase...")
                                env.reset()
                                showcase(env, best_brain)
                                env.reset()
                            except Exception as e:
                                print("Load failed:", e)
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); sys.exit(0)

        # Evaluate
        env.reset()
        evaluate_population(env, pop, draw=env.draw)
        # Allow toggling draw back on between generations
        if not env.draw and pygame.get_init():
            # check key state to re-enable drawing
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                env.draw = True

        # Track best
        gen_best = max(pop, key=lambda i: i.fitness)
        if gen_best.fitness > best_fitness:
            best_fitness = gen_best.fitness
            best_brain = gen_best.brain.copy()
            print(f"Gen {current_generation[0]} new best fitness: {best_fitness:.1f}")
            # autosave best occasionally
            try:
                best_brain.save(SAVE_PATH)
            except Exception:
                pass

        # Next generation
        pop = make_next_generation(pop)
        current_generation[0] += 1

    print("Training complete. Running showcase with best genome...")
    env.reset()
    showcase(env, best_brain)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)

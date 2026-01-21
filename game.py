import pygame
import numpy as np
from enum import Enum
from collections import namedtuple
from modules.map_generator import MapGenerator
from bots.bot03 import Bot
from modules.button import Button
import pygame.mixer as mixer


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SIZE = (800, 600)
AQUA_MARINE = (127, 255, 212)
CRIMSON = (220, 20, 60)
SNOW = (255, 250, 250)
GHOST_WHITE = (248, 248, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)
BROWN = (255, 255, 0)
ORANGE = (255, 140, 0)
ORANGERED = (255, 69, 0)

BLOCK_SIZE = 50


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SlideOrDie:
    def __init__(self, font: pygame.font.Font, bot_randomness: float, speed: int, chase: bool, delay: int, w=1600, h=900):
        self.chase = chase
        self.speed = speed
        self.delay = delay
        self.bot_randomness = bot_randomness
        self.font = font
        self.h = h
        self.w = w

        self.run = True
        generator = MapGenerator()
        self.map = generator.map
        self.spawn_point = generator.spawn_point
        self.enemy_spawn = generator.enemy_spawn

        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Slide or Die")
        self.clock = pygame.time.Clock()

        self.pause_button = Button(5, 5, pygame.image.load("images_and_fonts/pause2.png"), 1)
        self.reset_button = Button(60, 5, pygame.image.load("images_and_fonts/reset2.png"), 1)
        self.menu = Button(170, 5, pygame.image.load("images_and_fonts/menu2.png"), 1)
        self.cherry = pygame.image.load("images_and_fonts/cherry2.png")

        self.reset()
        self.bot = Bot(self.map, randomness=self.bot_randomness)
        self.quit = False

        mixer.init()
        self.click_sound = mixer.Sound("sound_and_music/click.mp3")
        self.slide = mixer.Sound("sound_and_music/slide.mp3")
        self.eat_sound = mixer.Sound("sound_and_music/money.wav")
        self.win_sound = mixer.Sound("sound_and_music/win.mp3")
        self.lose_sound = mixer.Sound("sound_and_music/lose.mp3")
        self.enemy_step_sound = mixer.Sound("sound_and_music/enemy_step1.mp3")
        self.enemy_eat_sound = mixer.Sound("sound_and_music/enemy_eat.mp3")
        self.die_sound = mixer.Sound("sound_and_music/death.mp3")
        self.sound_limit = 0 if self.delay > 3 else 2
        self.pause_music = mixer.Sound("sound_and_music/Ensemble - gbrysvg (No Copyright Music) _ Release Preview (128 kbps)2.mp3")
        self.background_music = mixer.Sound("sound_and_music/Rasta Vibes - Dave Osorio (No Copyright Music) _ Release Preview.mp3")
        self.background_music.play(-1)
        self.music_started = True

    def reset(self):
        self.paused = False
        self.enemy_direction = 0
        self.timer = 0
        self.direction = Direction.DOWN
        self.head = Point(self.spawn_point.x * BLOCK_SIZE, self.spawn_point.y * BLOCK_SIZE)
        self.enemy = Point(self.enemy_spawn.x * BLOCK_SIZE, self.enemy_spawn.y * BLOCK_SIZE)
        self.score = 0
        self.enemy_score = 0
        self.food = None
        self.place_food()
        self.enemy_steps = 0

    def place_food(self):
        x = np.random.randint(0, self.map.shape[0])
        y = np.random.randint(0, self.map.shape[1])
        while self.map[x, y]:
            x = np.random.randint(0, self.map.shape[0])
            y = np.random.randint(0, self.map.shape[1])
        
        self.food = Point(x * BLOCK_SIZE, y * BLOCK_SIZE)

    def respawn(self):
        x = np.random.randint(0, self.map.shape[0])
        y = np.random.randint(0, self.map.shape[1])
        while self.map[x, y] or Point(x, y) == self.enemy:
            x = np.random.randint(0, self.map.shape[0])
            y = np.random.randint(0, self.map.shape[1])

        self.head = Point(x * BLOCK_SIZE, y * BLOCK_SIZE)

    def controls(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
                self.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.direction = Direction.UP
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                if not self.paused:
                    self.slide.play()

    def play_step(self):
        self.controls()
        if not self.paused:
            self.timer += 1

            if not self._is_collision(self._move(check=True)):
                self._move()

            if self.enemy == self.food and not self.chase:
                self.place_food()
                self.enemy_score += 1
                self.enemy_eat_sound.play()
            elif self.enemy == self.head and self.chase:
                self.enemy_score += 1
                self.respawn()
                self.die_sound.play()

            if self.timer >= self.delay:
                self.timer = 0
                target = self.head if self.chase else self.food
                bot_action = self.bot.forward(self.enemy, target)
                bot_step = self._move(check=True,
                                      target=self.enemy,
                                      direction=bot_action)
                if not self._is_collision(bot_step):
                    self._move(target=self.enemy, direction=bot_action)
                    if bot_action != self.enemy_direction:
                        self.enemy_steps += 1
                        if self.enemy_steps >= self.sound_limit:
                            self.enemy_steps = 0
                            self.enemy_step_sound.play()
                            self.enemy_direction = bot_action

            if self.food == self.head:
                self.place_food()
                self.score += 1
                self.eat_sound.play()

        if self.score == 10:
            self.win_sound.play()
        if self.enemy_score == 10:
            self.lose_sound.play()

        self._update_ui()
        self.clock.tick(self.speed)

    def draw_map(self):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j]:
                    pygame.draw.rect(self.screen, BLUE, (i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    def _is_collision(self, pt):
        rect_head = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j]:
                    border = pygame.Rect(i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    if border.colliderect(rect_head):
                        return True

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return False

    def _move(self, check=False, target=None, direction=None):
        if target is None:
            x = self.head.x
            y = self.head.y
        else:
            x = target.x
            y = target.y

        if direction is None:
            direction = self.direction
        else:
            dirs = [Direction.RIGHT,
                    Direction.LEFT,
                    Direction.UP,
                    Direction.DOWN]
            direction = dirs[direction - 1]

        if direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        else:
            x += BLOCK_SIZE

        if not check:
            if target is None:
                self.head = Point(x, y)
            else:
                self.enemy = Point(x, y)
        return Point(x, y)

    def _update_ui(self):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.enemy.x, self.enemy.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, GREEN, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        self.screen.blit(self.cherry, (self.food.x, self.food.y))
        self.draw_map()

        if self.pause_button.draw(self.screen):
            self.click_sound.play()
            self.paused = False if self.paused else True
            self.pause_music.play(-1)
        if self.paused:
            self.music_started = False
            self.background_music.stop()
            if self.reset_button.draw(self.screen):
                self.reset()
                self.click_sound.play()
            if self.menu.draw(self.screen):
                self.click_sound.play()
                self.run = False
        else:
            self.pause_music.stop()
            if not self.music_started:
                self.background_music.play()
                self.music_started = True


        pygame.draw.rect(self.screen, GREEN, (int(self.w / 2 - 177), 0, 196, 50))
        pygame.draw.rect(self.screen, RED, (int(self.w / 2 + 15), 0, 196, 50))
        score = self.font.render(f"Score: {self.score}", True, WHITE, GREEN)
        self.screen.blit(score, (int(self.w / 2 - 172), 0))
        score = self.font.render(f"Enemy: {self.enemy_score}", True, WHITE, RED)
        self.screen.blit(score, (int(self.w / 2 + 15), 0))

        pygame.display.flip()

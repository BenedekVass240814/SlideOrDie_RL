import pygame
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


class Menu:
    def __init__(self, bot_randomness, delay, chase, speed, difficulty,
                 w=1600, h=900):
        self.h = h
        self.w = w
        self.run = True
        self.start_countdown = False
        self.quit_delay = 80
        self.difficulty = difficulty
        self.speed = speed

        self.background = pygame.image.load("images_and_fonts/background.png")
        self.slide = pygame.image.load("images_and_fonts/slide.png")
        self.press = pygame.image.load("images_and_fonts/press.png")
        self.chase_button = Button(BLOCK_SIZE * 18, BLOCK_SIZE * 1,
                                   pygame.image.load("images_and_fonts/chase2.png"), 1.5)
        self.speed_button = Button(BLOCK_SIZE * 18, BLOCK_SIZE * 3,
                                   pygame.image.load("images_and_fonts/speed2.png"), 1.5)
        self.difficulty_beginner = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 1,
                                          pygame.image.load("images_and_fonts/beginner2.png"), 1.5)
        self.difficulty_easy = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 3,
                                      pygame.image.load("images_and_fonts/easy2.png"), 1.5)
        self.difficulty_normal = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 5,
                                        pygame.image.load("images_and_fonts/normal2.png"), 1.5)
        self.difficulty_hard = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 7,
                                      pygame.image.load("images_and_fonts/hard2.png"), 1.5)
        self.difficulty_expert = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 9,
                                        pygame.image.load("images_and_fonts/expert2.png"), 1.5)
        self.difficulty_grandmaster = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 11,
                                             pygame.image.load("images_and_fonts/grandmaster2.png"), 1.5)
        self.difficulty_legend = Button(BLOCK_SIZE * 25, BLOCK_SIZE * 13,
                                        pygame.image.load("images_and_fonts/legend2.png"), 1.5)

        image = pygame.image.load("images_and_fonts/or_title.png")
        width = image.get_width()
        height = image.get_height()
        scale = 0.8
        self.or_title = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        image = pygame.image.load("images_and_fonts/die_title2.png")
        width = image.get_width()
        height = image.get_height()
        scale = 0.8
        self.die_title = pygame.transform.scale(image, (int(width * scale), int(height * scale)))

        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Slide or Die")
        self.clock = pygame.time.Clock()
        self.bot_randomness = bot_randomness
        self.delay = delay
        self.chase = chase
        self.quit = False
        self.max_score = 10

        self.title_font2 = pygame.font.Font("images_and_fonts/Kanit-Bold.ttf", 100)
        self.title_font3 = pygame.font.Font("images_and_fonts/DancingScript-VariableFont_wght.ttf", 170)
        self.basic_font = pygame.font.Font("images_and_fonts/Kanit-Bold.ttf", 50)

        mixer.init()
        self.click_sound = mixer.Sound("sound_and_music/click.mp3")
        self.beginner_start_sound = mixer.Sound("sound_and_music/beginner_start_sound.mp3")
        self.expert_start_sound = mixer.Sound("sound_and_music/expert+_start_sound.mp3")
        mixer.music.load("sound_and_music/Natural Space 20 - Dave Osorio (No Copyright Music) _ Release Preview.mp3")
        mixer.music.play(-1)

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
                self.quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    self.start_countdown = True
                    mixer.music.stop()
                    if self.difficulty > 3:
                        self.expert_start_sound.play()
                    else:
                        self.beginner_start_sound.play()

        if self.start_countdown:
            self.quit_delay -= 1
        if self.quit_delay <= 0:
            self.run = False

        self._update_ui()

        self.clock.tick(60)

    def get_settings(self):
        return self.bot_randomness, self.delay, self.chase, self.max_score, self.quit, self.speed, self.difficulty

    def _update_ui(self):
        self.screen.fill(WHITE)
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.press, (BLOCK_SIZE * 13.5, BLOCK_SIZE * 11.9))

        start = self.title_font3.render("Space", True, GREEN, BLACK)
        self.screen.blit(start, (BLOCK_SIZE * 13, BLOCK_SIZE * 14))

        pygame.draw.rect(self.screen, (self.speed * 12, 320 - self.speed * 15, 0),
                         (BLOCK_SIZE * 17.9, BLOCK_SIZE * 2.9,
                         BLOCK_SIZE * 6.2, BLOCK_SIZE * 1.7))

        if self.speed_button.draw(self.screen):
            self.click_sound.play()
            if self.speed == 20:
                self.speed = 8
            else:
                self.speed += 1

        color = GREEN if self.chase else RED
        pygame.draw.rect(self.screen, color, (BLOCK_SIZE * 17.9, BLOCK_SIZE * 0.9,
                                              BLOCK_SIZE * 6.2, BLOCK_SIZE * 1.7))

        self.screen.blit(self.slide, (BLOCK_SIZE, 10))
        self.screen.blit(self.or_title, (50, BLOCK_SIZE * 7.8))
        self.screen.blit(self.die_title, (50, BLOCK_SIZE * 12))

        pygame.draw.rect(self.screen, RED, (BLOCK_SIZE * 24.9, BLOCK_SIZE * 0.9 + BLOCK_SIZE * self.difficulty * 2,
                                            BLOCK_SIZE * 6.2, BLOCK_SIZE * 1.7))

        if self.chase_button.draw(self.screen):
            self.click_sound.play()
            self.chase = False if self.chase else True

        if self.difficulty_beginner.draw(self.screen):
            self.click_sound.play()
            self.delay = 5
            self.bot_randomness = 0.5
            self.difficulty = 0
        if self.difficulty_easy.draw(self.screen):
            self.click_sound.play()
            self.delay = 4
            self.bot_randomness = 0.4
            self.difficulty = 1
        if self.difficulty_normal.draw(self.screen):
            self.click_sound.play()
            self.delay = 4
            self.bot_randomness = 0.2
            self.difficulty = 2
        if self.difficulty_hard.draw(self.screen):
            self.click_sound.play()
            self.delay = 3
            self.bot_randomness = 0.2
            self.difficulty = 3
        if self.difficulty_expert.draw(self.screen):
            self.click_sound.play()
            self.delay = 2
            self.bot_randomness = 0.2
            self.difficulty = 4
        if self.difficulty_grandmaster.draw(self.screen):
            self.click_sound.play()
            self.delay = 1
            self.bot_randomness = 0.3
            self.difficulty = 5
        if self.difficulty_legend.draw(self.screen):
            self.click_sound.play()
            self.delay = 0
            self.bot_randomness = 0
            self.difficulty = 6

        pygame.display.flip()


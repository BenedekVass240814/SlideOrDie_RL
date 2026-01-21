import pygame
from game import SlideOrDie
from timeit import default_timer as timer
from menu import Menu

print("By: Hiper\n")

QUIT = False
BOT_RANDOMNESS = 0.5
DELAY = 5
CHASE = False
SPEED = 14
DIFFICULTY = 0

while not QUIT:
    pygame.init()
    menu = Menu(BOT_RANDOMNESS, DELAY, CHASE, SPEED, DIFFICULTY)
    while menu.run:
        menu.play_step()

    BOT_RANDOMNESS, DELAY, CHASE, MAX_SCORE, QUIT, SPEED, DIFFICULTY = menu.get_settings()
    pygame.quit()

    pygame.init()
    font = pygame.font.Font("images_and_fonts/FuzzyBubbles-Regular.ttf", 40)

    game = SlideOrDie(font, bot_randomness=BOT_RANDOMNESS, delay=DELAY,
                      chase=CHASE, speed=SPEED)

    start = timer()
    while game.run and not QUIT:
        game.play_step()
        QUIT = game.quit
        if game.score >= MAX_SCORE or game.enemy_score >= MAX_SCORE:
            end = timer()
            f = open("match_history.txt", "a", encoding="UTF-8")
            avg = 0 if game.score == 0 else (end - start) / game.score
            avg_e = 0 if game.enemy_score == 0 else (end - start) / game.enemy_score
            f.write(f"Settings: Speed: {SPEED} Max score: {MAX_SCORE}\n"
                    f"Randomness: {BOT_RANDOMNESS} Delay: {DELAY} Chase: {CHASE}\n"
                    f"Time played: {end - start:1.2f} mp\n"
                    f"Score: {game.score}\n"
                    f"Average time per score: {avg:1.2f}\n"
                    f"Enemy score: {game.enemy_score}\n"
                    f"Average time per score: {avg_e:1.2f}\n\n")
            game.reset()
            start = timer()

    pygame.quit()

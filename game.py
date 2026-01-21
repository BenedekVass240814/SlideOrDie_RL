import pygame
from game_core import SlideOrDieCore, Direction, BLOCK_SIZE
import pygame.mixer as mixer


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class SlideOrDie:
    """
    Rendering wrapper around SlideOrDieCore.
    Handles display, sound, and user input.
    """
    
    def __init__(self, font: pygame.font.Font, bot_randomness: float, speed: int, 
                 chase: bool, delay: int, w=1600, h=900):
        self.speed = speed
        self.font = font
        self.w = w
        self.h = h
        
        # Initialize core game logic
        self.game_core = SlideOrDieCore(
            bot_randomness=bot_randomness,
            chase=chase,
            delay=delay,
            w=w,
            h=h
        )
        
        # Enable bot opponent
        from bots.bot03 import Bot
        self.game_core.enable_bot(Bot)
        
        # UI setup
        self.run = True
        self.quit = False
        self.paused = False
        
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Slide or Die")
        self.clock = pygame.time.Clock()
        
        # Load UI elements
        from modules.button import Button
        self.pause_button = Button(5, 5, pygame.image.load("images_and_fonts/pause2.png"), 1)
        self.reset_button = Button(60, 5, pygame.image.load("images_and_fonts/reset2.png"), 1)
        self.menu = Button(170, 5, pygame.image.load("images_and_fonts/menu2.png"), 1)
        self.cherry = pygame.image.load("images_and_fonts/cherry2.png")
        
        # Sound setup
        mixer.init()
        self.click_sound = mixer.Sound("sound_and_music/click.mp3")
        self.slide = mixer.Sound("sound_and_music/slide.mp3")
        self.eat_sound = mixer.Sound("sound_and_music/money.wav")
        self.win_sound = mixer.Sound("sound_and_music/win.mp3")
        self.lose_sound = mixer.Sound("sound_and_music/lose.mp3")
        self.enemy_step_sound = mixer.Sound("sound_and_music/enemy_step1.mp3")
        self.enemy_eat_sound = mixer.Sound("sound_and_music/enemy_eat.mp3")
        self.die_sound = mixer.Sound("sound_and_music/death.mp3")
        self.pause_music = mixer.Sound("sound_and_music/Ensemble - gbrysvg (No Copyright Music) _ Release Preview (128 kbps)2.mp3")
        self.background_music = mixer.Sound("sound_and_music/Rasta Vibes - Dave Osorio (No Copyright Music) _ Release Preview.mp3")
        
        self.background_music.play(-1)
        self.music_started = True
        self.sound_limit = 0 if delay > 3 else 2
        self.enemy_steps = 0
    
    @property
    def score(self):
        return self.game_core.score
    
    @property
    def enemy_score(self):
        return self.game_core.enemy_score
    
    def reset(self):
        """Reset game state."""
        self.game_core.reset()
        self.paused = False
        self.enemy_steps = 0
    
    def controls(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
                self.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.game_core.set_direction(Direction.UP)
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.game_core.set_direction(Direction.LEFT)
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.game_core.set_direction(Direction.DOWN)
                if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.game_core.set_direction(Direction.RIGHT)
                if not self.paused:
                    self.slide.play()
    
    def play_step(self):
        """Execute one game step with rendering."""
        self.controls()
        
        if not self.paused:
            # Execute game logic
            events = self.game_core.step()
            
            # Play sounds based on events
            if events['player_collected']:
                self.eat_sound.play()
            
            if events['enemy_collected']:
                self.enemy_eat_sound.play()
            
            if events['player_caught']:
                self.die_sound.play()
            
            # Play enemy step sound
            if self.game_core.timer == 0:  # Just moved
                self.enemy_steps += 1
                if self.enemy_steps >= self.sound_limit:
                    self.enemy_steps = 0
                    self.enemy_step_sound.play()
            
            # Win/lose sounds
            if self.game_core.score == 10:
                self.win_sound.play()
            if self.game_core.enemy_score == 10:
                self.lose_sound.play()
        
        self._update_ui()
        self.clock.tick(self.speed)
    
    def draw_map(self):
        """Draw the game map."""
        map_data = self.game_core.map
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                if map_data[i, j]:
                    pygame.draw.rect(self.screen, BLUE, 
                                   (i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    
    def _update_ui(self):
        """Update display."""
        self.screen.fill(WHITE)
        
        state = self.game_core.get_state()
        
        # Draw game objects
        pygame.draw.rect(self.screen, RED, 
                        pygame.Rect(state['enemy'].x, state['enemy'].y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, GREEN, 
                        pygame.Rect(state['head'].x, state['head'].y, BLOCK_SIZE, BLOCK_SIZE))
        self.screen.blit(self.cherry, (state['food'].x, state['food'].y))
        self.draw_map()
        
        # Handle pause button
        if self.pause_button.draw(self.screen):
            self.click_sound.play()
            self.paused = not self.paused
            if self.paused:
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
        
        # Draw score
        pygame.draw.rect(self.screen, GREEN, (int(self.w / 2 - 177), 0, 196, 50))
        pygame.draw.rect(self.screen, RED, (int(self.w / 2 + 15), 0, 196, 50))
        score = self.font.render(f"Score: {state['score']}", True, WHITE, GREEN)
        self.screen.blit(score, (int(self.w / 2 - 172), 0))
        score = self.font.render(f"Enemy: {state['enemy_score']}", True, WHITE, RED)
        self.screen.blit(score, (int(self.w / 2 + 15), 0))
        
        pygame.display.flip()
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_core import SlideOrDieCore, Direction, Point, BLOCK_SIZE
import pygame
import os


class SlideOrDieEnv(gym.Env):
    """
    Gymnasium environment for Slide or Die game using the core logic.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', None], 'render_fps': 60}
    
    def __init__(self, render_mode=None, bot_randomness=0.5, speed=20, 
                 chase=False, delay=5, max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Game parameters
        self.bot_randomness = bot_randomness
        self.speed = speed
        self.chase = chase
        self.delay = delay
        
        # Initialize pygame only if rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.cherry = None
        
        if self.render_mode == 'human':
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((1600, 900))
            pygame.display.set_caption("Slide or Die - RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("images_and_fonts/FuzzyBubbles-Regular.ttf", 40)
            self.cherry = pygame.image.load("images_and_fonts/cherry2.png")
        else:
            # Headless mode - no display
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            if not pygame.get_init():
                pygame.init()
        
        # Initialize core game (no rendering, no sounds)
        self.game = SlideOrDieCore(
            bot_randomness=bot_randomness,
            chase=chase,
            delay=delay
        )
        
        # Enable bot opponent
        from bots.bot03 import Bot
        self.game.enable_bot(Bot)
        
        # Define action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        self.grid_width = self.game.map.shape[0]
        self.grid_height = self.game.map.shape[1]
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, self.grid_height, self.grid_width),
            dtype=np.float32
        )
        
        # Track previous scores for reward calculation
        self.prev_player_score = 0
        self.prev_enemy_score = 0
        
    def _get_observation(self):
        """Convert game state to observation."""
        obs = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        
        state = self.game.get_state()
        
        # Channel 0: Walls
        obs[0] = state['map'].T
        
        # Channel 1: Player position
        player_grid_x = int(state['head'].x / BLOCK_SIZE)
        player_grid_y = int(state['head'].y / BLOCK_SIZE)
        if 0 <= player_grid_x < self.grid_width and 0 <= player_grid_y < self.grid_height:
            obs[1, player_grid_y, player_grid_x] = 1
        
        # Channel 2: Enemy position
        enemy_grid_x = int(state['enemy'].x / BLOCK_SIZE)
        enemy_grid_y = int(state['enemy'].y / BLOCK_SIZE)
        if 0 <= enemy_grid_x < self.grid_width and 0 <= enemy_grid_y < self.grid_height:
            obs[2, enemy_grid_y, enemy_grid_x] = 1
        
        # Channel 3: Food position
        food_grid_x = int(state['food'].x / BLOCK_SIZE)
        food_grid_y = int(state['food'].y / BLOCK_SIZE)
        if 0 <= food_grid_x < self.grid_width and 0 <= food_grid_y < self.grid_height:
            obs[3, food_grid_y, food_grid_x] = 1
        
        return obs
    
    def _action_to_direction(self, action):
        """Convert action number to Direction enum."""
        action_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
        }
        return action_map[action]
    
    def _calculate_reward(self, events):
        """Calculate reward based on game events."""
        reward = 0.0
        
        # Reward for collecting food
        if events['player_collected']:
            reward += 10.0
            self.prev_player_score = self.game.score
        
        # Penalty for enemy scoring
        if events['enemy_collected'] or events['player_caught']:
            if events['player_caught']:
                reward -= 15.0  # Bigger penalty for being caught
            else:
                reward -= 10.0
            self.prev_enemy_score = self.game.enemy_score
        
        # Small time penalty
        reward -= 0.01
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.game.reset()
        self.current_step = 0
        self.prev_player_score = 0
        self.prev_enemy_score = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step."""
        self.current_step += 1
        
        # Convert action to direction and execute
        direction = self._action_to_direction(action)
        events = self.game.step(player_action=direction)
        
        # Calculate reward
        reward = self._calculate_reward(events)
        
        # Check termination
        terminated = events['done']
        if terminated:
            if self.game.score >= 10:
                reward += 100.0  # Win bonus
            elif self.game.enemy_score >= 10:
                reward -= 100.0  # Loss penalty
        
        # Check truncation
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Info
        info = {
            'player_score': self.game.score,
            'enemy_score': self.game.enemy_score,
            'step': self.current_step
        }
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the game state."""
        if self.render_mode == 'human' and self.screen is not None:
            WHITE = (255, 255, 255)
            RED = (255, 0, 0)
            GREEN = (0, 255, 0)
            BLUE = (0, 0, 255)
            
            self.screen.fill(WHITE)
            
            state = self.game.get_state()
            
            # Draw map
            for i in range(state['map'].shape[0]):
                for j in range(state['map'].shape[1]):
                    if state['map'][i, j]:
                        pygame.draw.rect(self.screen, BLUE, 
                                       (i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            
            # Draw entities
            pygame.draw.rect(self.screen, RED, 
                           pygame.Rect(state['enemy'].x, state['enemy'].y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, GREEN, 
                           pygame.Rect(state['head'].x, state['head'].y, BLOCK_SIZE, BLOCK_SIZE))
            self.screen.blit(self.cherry, (state['food'].x, state['food'].y))
            
            # Draw scores
            score_text = self.font.render(f"Score: {state['score']}", True, WHITE, GREEN)
            enemy_text = self.font.render(f"Enemy: {state['enemy_score']}", True, WHITE, RED)
            self.screen.blit(score_text, (20, 20))
            self.screen.blit(enemy_text, (20, 70))
            
            pygame.display.flip()
            self.clock.tick(self.speed)
    
    def close(self):
        """Clean up."""
        if self.screen:
            pygame.quit()
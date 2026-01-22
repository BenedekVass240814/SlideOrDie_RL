import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_core import SlideOrDieCore, Direction, Point, BLOCK_SIZE
import pygame
import os
import math

class SmartSlideOrDieEnv(gym.Env):
    """
    Improved environment with Feature-based observations and Reward Shaping.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, bot_randomness=0.3, speed=60, 
                 chase=False, delay=5, max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Initialize game
        self.game = SlideOrDieCore(
            bot_randomness=bot_randomness,
            chase=chase,
            delay=delay
        )
        
        # Add bot
        from bots.bot03 import Bot
        self.game.enable_bot(Bot)
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: vector of 12 features
        # [food_dir_x, food_dir_y, enemy_dir_x, enemy_dir_y, 
        #  wall_dist_up, wall_dist_down, wall_dist_left, wall_dist_right,
        #  blocked_up, blocked_down, blocked_left, blocked_right]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Rendering setup
        self.screen = None
        self.clock = None
        self.speed = speed
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((1600, 900))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _cast_ray(self, start_pos, direction_enum):
        """Calculates normalized distance to wall in a specific direction."""
        x, y = start_pos.x, start_pos.y
        dist = 0
        
        # Move step by step until collision
        while True:
            if direction_enum == Direction.UP: y -= BLOCK_SIZE
            elif direction_enum == Direction.DOWN: y += BLOCK_SIZE
            elif direction_enum == Direction.LEFT: x -= BLOCK_SIZE
            elif direction_enum == Direction.RIGHT: x += BLOCK_SIZE
            
            # Check boundaries
            if x < 0 or x >= self.game.w or y < 0 or y >= self.game.h:
                break
            
            # Check map walls
            grid_x, grid_y = int(x // BLOCK_SIZE), int(y // BLOCK_SIZE)
            if self.game.map[grid_x, grid_y]:
                break
                
            dist += 1
            
        # Normalize distance (max possible distance is roughly diagonal of map)
        max_dist = max(self.game.w, self.game.h) // BLOCK_SIZE
        return min(dist / max_dist, 1.0)

    def _is_blocked(self, pos, direction_enum):
        """Checks if the immediate next tile is blocked."""
        x, y = pos.x, pos.y
        if direction_enum == Direction.UP: y -= BLOCK_SIZE
        elif direction_enum == Direction.DOWN: y += BLOCK_SIZE
        elif direction_enum == Direction.LEFT: x -= BLOCK_SIZE
        elif direction_enum == Direction.RIGHT: x += BLOCK_SIZE
        
        # Bounds
        if x < 0 or x >= self.game.w or y < 0 or y >= self.game.h:
            return 1.0
        
        # Walls
        grid_x, grid_y = int(x // BLOCK_SIZE), int(y // BLOCK_SIZE)
        if self.game.map[grid_x, grid_y]:
            return 1.0
            
        return 0.0

    def _get_observation(self):
        state = self.game.get_state()
        head = state['head']
        food = state['food']
        enemy = state['enemy']
        
        # 1. Relative direction to Food (Normalized)
        food_dx = (food.x - head.x) / self.game.w
        food_dy = (food.y - head.y) / self.game.h
        
        # 2. Relative direction to Enemy (Normalized)
        enemy_dx = (enemy.x - head.x) / self.game.w
        enemy_dy = (enemy.y - head.y) / self.game.h
        
        # 3. Wall Distances (Raycasting)
        dist_up = self._cast_ray(head, Direction.UP)
        dist_down = self._cast_ray(head, Direction.DOWN)
        dist_left = self._cast_ray(head, Direction.LEFT)
        dist_right = self._cast_ray(head, Direction.RIGHT)
        
        # 4. Immediate Blockage (Binary)
        blk_up = self._is_blocked(head, Direction.UP)
        blk_down = self._is_blocked(head, Direction.DOWN)
        blk_left = self._is_blocked(head, Direction.LEFT)
        blk_right = self._is_blocked(head, Direction.RIGHT)
        
        obs = np.array([
            food_dx, food_dy,
            enemy_dx, enemy_dy,
            dist_up, dist_down, dist_left, dist_right,
            blk_up, blk_down, blk_left, blk_right
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game.reset()
        self.current_step = 0
        self.prev_dist_to_food = self._get_distance(self.game.head, self.game.food)
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # Map action index to Direction
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        move_dir = dirs[action]
        
        # Game Step
        events = self.game.step(player_action=move_dir)
        
        # --- REWARD SHAPING ---
        reward = 0
        
        # 1. Base step penalty (encourage speed)
        reward -= 0.01
        
        # 2. Distance Shaping (Reward for getting closer to food)
        current_dist = self._get_distance(self.game.head, self.game.food)
        if current_dist < self.prev_dist_to_food:
            reward += 0.1  # Good boy, getting closer
        else:
            reward -= 0.15 # Bad boy, moving away or hitting wall (stuck)
        self.prev_dist_to_food = current_dist
        
        # 3. Event Rewards
        if events['player_collected']:
            reward += 20.0
            self.prev_dist_to_food = self._get_distance(self.game.head, self.game.food) # Reset dist tracker
            
        if events['enemy_collected']:
            reward -= 5.0
            self.prev_dist_to_food = self._get_distance(self.game.head, self.game.food)
            
        if events['player_caught']:
            reward -= 20.0
            
        # 4. Game Over Logic
        terminated = events['done']
        if terminated:
            if self.game.score >= 10:
                reward += 50.0
            elif self.game.enemy_score >= 10:
                reward -= 50.0
                
        truncated = self.current_step >= self.max_steps
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_observation(), reward, terminated, truncated, {'score': self.game.score}

    def render(self):
        if self.screen is None: return
        self.screen.fill((255, 255, 255))
        
        state = self.game.get_state()
        
        # Draw Map
        for x in range(state['map'].shape[0]):
            for y in range(state['map'].shape[1]):
                if state['map'][x, y]:
                    pygame.draw.rect(self.screen, (0, 0, 100), 
                                   (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Food
        pygame.draw.circle(self.screen, (200, 0, 0), 
                         (state['food'].x + BLOCK_SIZE//2, state['food'].y + BLOCK_SIZE//2), BLOCK_SIZE//2)
        
        # Draw Enemy
        pygame.draw.rect(self.screen, (255, 0, 0), 
                       (state['enemy'].x, state['enemy'].y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Player
        pygame.draw.rect(self.screen, (0, 255, 0), 
                       (state['head'].x, state['head'].y, BLOCK_SIZE, BLOCK_SIZE))
                       
        pygame.display.flip()
        self.clock.tick(self.speed)
        
    def close(self):
        if self.screen: pygame.quit()
"""
Gymnasium environment wrapper for Slide or Die.
Uses compact feature-based observations for efficient learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import pygame
import os
import sys

# Add parent and current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_core import SlideOrDieCore, Action, Point, BLOCK_SIZE
from config import EnvConfig, RewardConfig, DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG


class SlideOrDieEnv(gym.Env):
    """
    Gymnasium environment for Slide or Die with COMPACT observations.
    
    Key differences from the original RL setup:
    1. Feature-based observation instead of full grid image
    2. Includes relative positions, distances, and local obstacle info
    3. Much smaller observation space = faster training
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}
    
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        super().__init__()
        
        self.env_config = env_config or DEFAULT_ENV_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.render_mode = render_mode
        
        # Initialize core game
        self.game = SlideOrDieCore(self.env_config, self.reward_config)
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: compact feature vector
        # Features:
        # - Player grid position (normalized): 2
        # - Food relative position (normalized): 2
        # - Enemy relative position (normalized): 2
        # - Distance to food (normalized): 1
        # - Distance to enemy (normalized): 1
        # - Local obstacles (4 directions): 4
        # - Player direction (one-hot): 4
        # - Score difference (normalized): 1
        # Total: 17 features
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )
        
        # Rendering setup
        self.screen = None
        self.clock = None
        self.font = None
        self.cherry = None
        
        if self.render_mode == "human":
            self._init_rendering()
    
    def _init_rendering(self):
        """Initialize pygame for rendering."""
        if not pygame.get_init():
            pygame.init()
        
        self.screen = pygame.display.set_mode((self.game.w, self.game.h))
        pygame.display.set_caption("Slide or Die - RL Training")
        self.clock = pygame.time.Clock()
        
        # Load assets
        font_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "images_and_fonts", "FuzzyBubbles-Regular.ttf"
        )
        cherry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "images_and_fonts", "cherry2.png"
        )
        
        if os.path.exists(font_path):
            self.font = pygame.font.Font(font_path, 40)
        else:
            self.font = pygame.font.Font(None, 40)
        
        if os.path.exists(cherry_path):
            self.cherry = pygame.image.load(cherry_path)
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert game state to compact feature vector.
        This is the KEY innovation - much better than full grid!
        """
        state = self.game._get_state()
        
        # Normalize positions to [-1, 1]
        max_x = self.game.w
        max_y = self.game.h
        
        player_x = state["player_pos"].x / max_x * 2 - 1
        player_y = state["player_pos"].y / max_y * 2 - 1
        
        # Relative positions (food and enemy relative to player)
        food_rel_x = (state["food_pos"].x - state["player_pos"].x) / max_x * 2
        food_rel_y = (state["food_pos"].y - state["player_pos"].y) / max_y * 2
        
        enemy_rel_x = (state["enemy_pos"].x - state["player_pos"].x) / max_x * 2
        enemy_rel_y = (state["enemy_pos"].y - state["player_pos"].y) / max_y * 2
        
        # Normalized distances
        max_dist = max_x + max_y
        dist_to_food = (abs(state["food_pos"].x - state["player_pos"].x) + 
                        abs(state["food_pos"].y - state["player_pos"].y)) / max_dist
        dist_to_enemy = (abs(state["enemy_pos"].x - state["player_pos"].x) + 
                         abs(state["enemy_pos"].y - state["player_pos"].y)) / max_dist
        
        # Local obstacles (1 if obstacle, 0 if clear)
        obstacles = self._get_local_obstacles(state["player_pos"])
        
        # Direction one-hot
        direction = np.zeros(4, dtype=np.float32)
        direction[state["player_direction"]] = 1.0
        
        # Score difference (normalized to [-1, 1])
        score_diff = (state["score"] - state["enemy_score"]) / self.env_config.win_score
        score_diff = np.clip(score_diff, -1.0, 1.0)
        
        # Combine all features
        obs = np.array([
            player_x, player_y,
            food_rel_x, food_rel_y,
            enemy_rel_x, enemy_rel_y,
            dist_to_food,
            dist_to_enemy,
            obstacles[0], obstacles[1], obstacles[2], obstacles[3],
            direction[0], direction[1], direction[2], direction[3],
            score_diff
        ], dtype=np.float32)
        
        return obs
    
    def _get_local_obstacles(self, pos: Point) -> np.ndarray:
        """Check for obstacles in each direction from current position."""
        obstacles = np.zeros(4, dtype=np.float32)
        
        bs = self.game.block_size
        directions = [
            (0, -bs),  # UP
            (0, bs),   # DOWN
            (-bs, 0),  # LEFT
            (bs, 0),   # RIGHT
        ]
        
        for i, (dx, dy) in enumerate(directions):
            new_pos = Point(pos.x + dx, pos.y + dy)
            if not self.game._is_valid_move(new_pos):
                obstacles[i] = 1.0
        
        return obstacles
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.game.reset()
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return new state."""
        _, reward, terminated, truncated, info = self.game.step(action)
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current game state."""
        if self.render_mode == "human":
            if self.screen is None:
                self._init_rendering()
            
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
        
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()
    
    def _render_frame(self):
        """Render a single frame."""
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        
        self.screen.fill(WHITE)
        
        # Draw walls
        for i in range(self.game.map.shape[0]):
            for j in range(self.game.map.shape[1]):
                if self.game.map[i, j]:
                    pygame.draw.rect(
                        self.screen, BLUE,
                        (i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    )
        
        # Draw enemy
        pygame.draw.rect(
            self.screen, RED,
            (self.game.enemy_pos.x, self.game.enemy_pos.y, BLOCK_SIZE, BLOCK_SIZE)
        )
        
        # Draw player
        pygame.draw.rect(
            self.screen, GREEN,
            (self.game.player_pos.x, self.game.player_pos.y, BLOCK_SIZE, BLOCK_SIZE)
        )
        
        # Draw food
        if self.cherry:
            self.screen.blit(self.cherry, (self.game.food_pos.x, self.game.food_pos.y))
        else:
            pygame.draw.circle(
                self.screen, (255, 200, 0),
                (self.game.food_pos.x + BLOCK_SIZE // 2, 
                 self.game.food_pos.y + BLOCK_SIZE // 2),
                BLOCK_SIZE // 3
            )
        
        # Draw scores
        if self.font:
            score_text = self.font.render(
                f"Player: {self.game.score}  Enemy: {self.game.enemy_score}  "
                f"Steps: {self.game.steps}", True, (0, 0, 0)
            )
            self.screen.blit(score_text, (10, 10))
    
    def _get_rgb_array(self) -> np.ndarray:
        """Get current frame as RGB array."""
        if self.screen is None:
            # Create offscreen surface
            surface = pygame.Surface((self.game.w, self.game.h))
            self.screen = surface
            self._render_frame()
        else:
            self._render_frame()
        
        return np.transpose(
            pygame.surfarray.array3d(self.screen), 
            axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Register environment with gymnasium
def register_env():
    """Register the environment with gymnasium."""
    from gymnasium.envs.registration import register
    
    try:
        register(
            id="SlideOrDie-v1",
            entry_point="env:SlideOrDieEnv",
        )
    except Exception:
        pass  # Already registered


if __name__ == "__main__":
    # Quick test
    env = SlideOrDieEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

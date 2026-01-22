"""
Improved Gymnasium environment with:
1. Frame stacking for temporal memory
2. Better observations (velocity, enemy direction)
3. Improved reward shaping
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque
import pygame
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_core import SlideOrDieCore, Action, Point, BLOCK_SIZE
from config import EnvConfig, RewardConfig, DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG


class SlideOrDieEnvV2(gym.Env):
    """
    Improved environment with memory via frame stacking.
    
    Key improvements:
    1. Frame stacking - agent sees last N frames
    2. Velocity information - knows direction of movement
    3. Enemy direction - knows where enemy is heading
    4. Stuck detection - penalty for not moving
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}
    
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        map_config = None,  # MapConfig for fixed map training
        frame_stack: int = 4,  # Number of frames to stack
    ):
        super().__init__()
        
        self.env_config = env_config or DEFAULT_ENV_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.map_config = map_config
        self.render_mode = render_mode
        self.frame_stack = frame_stack
        
        # Initialize core game with map config
        self.game = SlideOrDieCore(self.env_config, self.reward_config, self.map_config)
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Single frame observation size
        # Features per frame:
        # - Player grid position (normalized): 2
        # - Food relative position (normalized): 2
        # - Enemy relative position (normalized): 2
        # - Distance to food (normalized): 1
        # - Distance to enemy (normalized): 1
        # - Local obstacles (4 directions): 4
        # - Player velocity (dx, dy from last frame): 2
        # - Enemy velocity (dx, dy from last move): 2
        # - Score difference (normalized): 1
        # - Steps remaining (normalized): 1
        # Total per frame: 18
        
        self.single_obs_size = 18
        
        # With frame stacking
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.single_obs_size * frame_stack,),
            dtype=np.float32
        )
        
        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=frame_stack)
        
        # Track previous positions for velocity calculation
        self.prev_player_pos = None
        self.prev_enemy_pos = None
        self.prev_action = None
        
        # Stuck detection
        self.position_history = deque(maxlen=10)
        self.stuck_counter = 0
        
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
        pygame.display.set_caption("Slide or Die - RL Training V2")
        self.clock = pygame.time.Clock()
        
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
    
    def _get_single_observation(self) -> np.ndarray:
        """Get observation for current frame."""
        state = self.game._get_state()
        
        max_x = self.game.w
        max_y = self.game.h
        
        # Player position (normalized)
        player_x = state["player_pos"].x / max_x * 2 - 1
        player_y = state["player_pos"].y / max_y * 2 - 1
        
        # Relative positions
        food_rel_x = (state["food_pos"].x - state["player_pos"].x) / max_x * 2
        food_rel_y = (state["food_pos"].y - state["player_pos"].y) / max_y * 2
        
        enemy_rel_x = (state["enemy_pos"].x - state["player_pos"].x) / max_x * 2
        enemy_rel_y = (state["enemy_pos"].y - state["player_pos"].y) / max_y * 2
        
        # Distances
        max_dist = max_x + max_y
        dist_to_food = (abs(state["food_pos"].x - state["player_pos"].x) + 
                        abs(state["food_pos"].y - state["player_pos"].y)) / max_dist
        dist_to_enemy = (abs(state["enemy_pos"].x - state["player_pos"].x) + 
                         abs(state["enemy_pos"].y - state["player_pos"].y)) / max_dist
        
        # Local obstacles
        obstacles = self._get_local_obstacles(state["player_pos"])
        
        # Player velocity (change from last frame)
        if self.prev_player_pos is not None:
            player_vel_x = (state["player_pos"].x - self.prev_player_pos.x) / BLOCK_SIZE
            player_vel_y = (state["player_pos"].y - self.prev_player_pos.y) / BLOCK_SIZE
        else:
            player_vel_x, player_vel_y = 0.0, 0.0
        
        # Enemy velocity
        if self.prev_enemy_pos is not None:
            enemy_vel_x = (state["enemy_pos"].x - self.prev_enemy_pos.x) / BLOCK_SIZE
            enemy_vel_y = (state["enemy_pos"].y - self.prev_enemy_pos.y) / BLOCK_SIZE
        else:
            enemy_vel_x, enemy_vel_y = 0.0, 0.0
        
        # Score difference
        score_diff = (state["score"] - state["enemy_score"]) / self.env_config.win_score
        score_diff = np.clip(score_diff, -1.0, 1.0)
        
        # Steps remaining (normalized)
        steps_remaining = 1.0 - (state["steps"] / self.env_config.max_steps)
        
        obs = np.array([
            player_x, player_y,
            food_rel_x, food_rel_y,
            enemy_rel_x, enemy_rel_y,
            dist_to_food,
            dist_to_enemy,
            obstacles[0], obstacles[1], obstacles[2], obstacles[3],
            player_vel_x, player_vel_y,
            enemy_vel_x, enemy_vel_y,
            score_diff,
            steps_remaining,
        ], dtype=np.float32)
        
        return obs
    
    def _get_stacked_observation(self) -> np.ndarray:
        """Get stacked observation from frame buffer."""
        # Fill buffer if needed
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.append(self._get_single_observation())
        
        # Concatenate all frames
        return np.concatenate(list(self.frame_buffer))
    
    def _get_local_obstacles(self, pos: Point) -> np.ndarray:
        """Check for obstacles in each direction."""
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
    
    def _check_stuck(self, pos: Point) -> bool:
        """Check if agent is stuck (not moving for several steps)."""
        self.position_history.append((pos.x, pos.y))
        
        if len(self.position_history) >= 10:
            # Check if all recent positions are the same
            unique_positions = set(self.position_history)
            if len(unique_positions) <= 2:
                return True
        
        return False
    
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
        
        # Reset tracking
        self.prev_player_pos = None
        self.prev_enemy_pos = None
        self.prev_action = None
        self.position_history.clear()
        self.stuck_counter = 0
        
        # Clear frame buffer and fill with initial observation
        self.frame_buffer.clear()
        initial_obs = self._get_single_observation()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(initial_obs.copy())
        
        obs = self._get_stacked_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with improved reward shaping."""
        # Store previous positions
        self.prev_player_pos = self.game.player_pos
        self.prev_enemy_pos = self.game.enemy_pos
        
        # Execute game step
        _, base_reward, terminated, truncated, info = self.game.step(action)
        
        # Use base reward directly - game_core already has good shaping
        reward = base_reward
        
        # Only add a small penalty for repeatedly not moving (stuck)
        moved = self.game.player_pos != self.prev_player_pos
        if not moved:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Small stuck penalty only after several consecutive stuck moves
        if self.stuck_counter > 10:
            reward -= 0.1
        
        self.prev_action = action
        
        # Update position history for diagnostics
        self._check_stuck(self.game.player_pos)
        
        # Update frame buffer
        self.frame_buffer.append(self._get_single_observation())
        
        obs = self._get_stacked_observation()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current game state."""
        if self.render_mode == "human":
            if self.screen is None:
                self._init_rendering()
            
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
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


if __name__ == "__main__":
    # Quick test
    env = SlideOrDieEnvV2(render_mode="human", frame_stack=4)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Single frame: 18 features x {env.frame_stack} frames = {18 * env.frame_stack}")
    
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

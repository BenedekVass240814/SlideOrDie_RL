"""
Core game logic without rendering - optimized for RL training.
This is a minimal, fast version of the game for training purposes.
"""

import numpy as np
from enum import IntEnum
from collections import namedtuple
from typing import Tuple, Optional, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.map_generator import MapGenerator


BLOCK_SIZE = 50


class Action(IntEnum):
    """Actions the agent can take."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


Point = namedtuple("Point", "x, y")


class SimpleOpponent:
    """
    Simple opponent types for curriculum learning.
    No pathfinding - the agent should learn to beat these first.
    """
    
    def __init__(self, opponent_type: str = "random", randomness: float = 0.5):
        self.opponent_type = opponent_type
        self.randomness = randomness
        self.last_action = None
    
    def get_action(self, enemy_pos: Point, target_pos: Point, 
                   valid_actions: list) -> int:
        """
        Get opponent action based on type.
        
        Args:
            enemy_pos: Current enemy position
            target_pos: Target position (food or player)
            valid_actions: List of valid actions (no wall collision)
        
        Returns:
            Action index (1-4 for compatibility with original game)
        """
        if not valid_actions:
            return 1  # Default to RIGHT
        
        if self.opponent_type == "random":
            return self._random_action(valid_actions)
        elif self.opponent_type == "simple":
            return self._simple_chase(enemy_pos, target_pos, valid_actions)
        else:
            return self._random_action(valid_actions)
    
    def _random_action(self, valid_actions: list) -> int:
        return np.random.choice(valid_actions)
    
    def _simple_chase(self, enemy_pos: Point, target_pos: Point,
                      valid_actions: list) -> int:
        """Simple greedy chase - move towards target with some randomness."""
        if np.random.random() < self.randomness:
            return self._random_action(valid_actions)
        
        # Calculate direction to target
        dx = target_pos.x - enemy_pos.x
        dy = target_pos.y - enemy_pos.y
        
        # Prioritize larger distance axis
        preferred = []
        if abs(dx) > abs(dy):
            if dx > 0 and 1 in valid_actions:  # RIGHT
                preferred.append(1)
            elif dx < 0 and 2 in valid_actions:  # LEFT
                preferred.append(2)
        else:
            if dy > 0 and 4 in valid_actions:  # DOWN
                preferred.append(4)
            elif dy < 0 and 3 in valid_actions:  # UP
                preferred.append(3)
        
        if preferred:
            return preferred[0]
        return self._random_action(valid_actions)


class SlideOrDieCore:
    """
    Minimal game core for RL training.
    Optimized for speed - no rendering, no sounds.
    """
    
    def __init__(self, env_config=None, reward_config=None, map_config=None):
        from config import EnvConfig, RewardConfig, DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG
        from wandb_config import MapConfig, DEFAULT_MAP_CONFIG
        
        self.env_config = env_config or DEFAULT_ENV_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.map_config = map_config or DEFAULT_MAP_CONFIG
        
        self.w = self.env_config.width
        self.h = self.env_config.height
        self.block_size = self.env_config.block_size
        
        # Grid dimensions
        self.grid_w = self.w // self.block_size
        self.grid_h = self.h // self.block_size
        
        # Map tracking
        self.episode_count = 0
        self.map_change_count = 0
        self.current_map_seed = self.map_config.map_seed
        
        # Generate initial map
        self._generate_map()
        
        # Opponent
        self.opponent = None
        if self.env_config.enable_opponent:
            self.opponent = SimpleOpponent(
                opponent_type=self.env_config.opponent_type,
                randomness=0.5
            )
        
        self.reset()
    
    def _generate_map(self, seed: int = None):
        """Generate a new map, optionally with a specific seed."""
        generator = MapGenerator(w=self.w, h=self.h, seed=seed or self.current_map_seed)
        self.map = generator.map
        self.spawn_point = generator.spawn_point
        self.enemy_spawn = generator.enemy_spawn
        
        # Precompute valid positions for faster collision checking
        self._precompute_valid_positions()
        
        self.map_change_count += 1
    
    def _precompute_valid_positions(self):
        """Precompute set of valid grid positions for O(1) collision checking."""
        self.valid_positions = set()
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                if not self.map[i, j]:
                    self.valid_positions.add((i, j))
    
    def reset(self, force_new_map: bool = False) -> Dict[str, Any]:
        """Reset game state and return initial observation.
        
        Args:
            force_new_map: Force generation of a new map regardless of settings
        """
        self.episode_count += 1
        
        # Check if we should generate a new map
        should_change_map = force_new_map
        
        if self.map_config.use_fixed_map:
            # Change map every N episodes
            if (self.episode_count % self.map_config.map_change_episodes == 0 
                and self.episode_count > 0):
                should_change_map = True
        else:
            # Random map every episode (original behavior)
            should_change_map = True
        
        if should_change_map:
            # Generate new random seed for variety
            new_seed = np.random.randint(0, 1000000) if self.current_map_seed is None else None
            self._generate_map(seed=new_seed)
        
        self.steps = 0
        self.timer = 0
        
        # Player state
        self.player_pos = Point(
            self.spawn_point.x * self.block_size,
            self.spawn_point.y * self.block_size
        )
        self.player_direction = Action.DOWN
        self.score = 0
        
        # Enemy state
        self.enemy_pos = Point(
            self.enemy_spawn.x * self.block_size,
            self.enemy_spawn.y * self.block_size
        )
        self.enemy_score = 0
        
        # Food
        self.food_pos = None
        self._place_food()
        
        # Track previous distance for reward shaping
        self._prev_dist_to_food = self._distance_to_food()
        
        return self._get_state()
    
    def _place_food(self):
        """Place food at random valid position."""
        valid_list = list(self.valid_positions)
        idx = np.random.randint(len(valid_list))
        pos = valid_list[idx]
        self.food_pos = Point(pos[0] * self.block_size, pos[1] * self.block_size)
    
    def _to_grid(self, pos: Point) -> Tuple[int, int]:
        """Convert pixel position to grid position."""
        return (pos.x // self.block_size, pos.y // self.block_size)
    
    def _is_valid_move(self, pos: Point) -> bool:
        """Check if position is valid (no wall, in bounds)."""
        gx, gy = pos.x // self.block_size, pos.y // self.block_size
        
        # Boundary check
        if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
            return False
        
        # Wall check using precomputed set
        return (gx, gy) in self.valid_positions
    
    def _get_new_position(self, pos: Point, action: int) -> Point:
        """Calculate new position after action."""
        x, y = pos.x, pos.y
        
        if action == Action.UP:
            y -= self.block_size
        elif action == Action.DOWN:
            y += self.block_size
        elif action == Action.LEFT:
            x -= self.block_size
        elif action == Action.RIGHT:
            x += self.block_size
        
        return Point(x, y)
    
    def _get_valid_actions(self, pos: Point) -> list:
        """Get list of valid actions from a position."""
        valid = []
        for action in [1, 2, 3, 4]:  # RIGHT, LEFT, UP, DOWN (original game format)
            new_pos = self._get_new_position_old_format(pos, action)
            if self._is_valid_move(new_pos):
                valid.append(action)
        return valid
    
    def _get_new_position_old_format(self, pos: Point, action: int) -> Point:
        """Get new position using original game action format (1-4)."""
        x, y = pos.x, pos.y
        if action == 1:  # RIGHT
            x += self.block_size
        elif action == 2:  # LEFT
            x -= self.block_size
        elif action == 3:  # UP
            y -= self.block_size
        elif action == 4:  # DOWN
            y += self.block_size
        return Point(x, y)
    
    def _distance_to_food(self) -> float:
        """Manhattan distance from player to food."""
        return abs(self.player_pos.x - self.food_pos.x) + \
               abs(self.player_pos.y - self.food_pos.y)
    
    def _distance_enemy_to_food(self) -> float:
        """Manhattan distance from enemy to food."""
        return abs(self.enemy_pos.x - self.food_pos.x) + \
               abs(self.enemy_pos.y - self.food_pos.y)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one game step.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        self.timer += 1
        reward = self.reward_config.time_penalty
        terminated = False
        truncated = False
        info = {"player_collected": False, "enemy_collected": False, "caught": False}
        
        # Store old distance for shaping
        old_dist = self._prev_dist_to_food
        
        # Move player
        new_pos = self._get_new_position(self.player_pos, action)
        if self._is_valid_move(new_pos):
            self.player_pos = new_pos
            self.player_direction = action
        else:
            reward += self.reward_config.wall_collision
        
        # Check food collection
        if self.player_pos == self.food_pos:
            self.score += 1
            reward += self.reward_config.food_collected
            self._place_food()
            info["player_collected"] = True
        
        # Distance-based reward shaping
        new_dist = self._distance_to_food()
        if new_dist < old_dist:
            reward += self.reward_config.closer_to_food
        elif new_dist > old_dist:
            reward += self.reward_config.further_from_food
        self._prev_dist_to_food = new_dist
        
        # Enemy turn (if enabled and timer allows)
        if self.opponent and self.timer >= self.env_config.opponent_delay:
            self.timer = 0
            target = self.player_pos if self.env_config.chase_mode else self.food_pos
            valid_actions = self._get_valid_actions(self.enemy_pos)
            
            enemy_action = self.opponent.get_action(
                self.enemy_pos, target, valid_actions
            )
            
            new_enemy_pos = self._get_new_position_old_format(self.enemy_pos, enemy_action)
            if self._is_valid_move(new_enemy_pos):
                self.enemy_pos = new_enemy_pos
            
            # Check enemy food collection (non-chase mode)
            if not self.env_config.chase_mode and self.enemy_pos == self.food_pos:
                self.enemy_score += 1
                reward += self.reward_config.enemy_collected
                self._place_food()
                self._prev_dist_to_food = self._distance_to_food()
                info["enemy_collected"] = True
            
            # Check if enemy caught player (chase mode)
            if self.env_config.chase_mode and self.enemy_pos == self.player_pos:
                self.enemy_score += 1
                reward += self.reward_config.caught_by_enemy
                self._respawn_player()
                info["caught"] = True
        
        # Check win/lose conditions based on win_mode
        episode_ended = False
        
        if self.env_config.win_mode == "first_to_n":
            # Original mode: first to reach win_score wins
            if self.score >= self.env_config.win_score:
                reward += self.reward_config.win_bonus
                terminated = True
                episode_ended = True
                info["won"] = True
            elif self.enemy_score >= self.env_config.win_score:
                reward += self.reward_config.lose_penalty
                terminated = True
                episode_ended = True
                info["won"] = False
        
        # Check max steps (for both modes)
        if self.steps >= self.env_config.max_steps and not episode_ended:
            truncated = True
            
            # In score_compare mode, determine winner at end of episode
            if self.env_config.win_mode == "score_compare":
                if self.score > self.enemy_score:
                    reward += self.reward_config.win_bonus
                    info["won"] = True
                elif self.score < self.enemy_score:
                    reward += self.reward_config.lose_penalty
                    info["won"] = False
                else:
                    # Draw - small penalty to encourage winning
                    reward += self.reward_config.draw_penalty
                    info["won"] = False  # Treat draw as not winning
                    info["draw"] = True
        
        # Store final scores in info
        info["player_score"] = self.score
        info["enemy_score"] = self.enemy_score
        info["steps"] = self.steps
        
        return self._get_state(), reward, terminated, truncated, info
    
    def _respawn_player(self):
        """Respawn player at random valid position."""
        valid_list = [p for p in self.valid_positions 
                      if Point(p[0] * self.block_size, p[1] * self.block_size) != self.enemy_pos]
        idx = np.random.randint(len(valid_list))
        pos = valid_list[idx]
        self.player_pos = Point(pos[0] * self.block_size, pos[1] * self.block_size)
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current game state as a dictionary."""
        return {
            "player_pos": self.player_pos,
            "enemy_pos": self.enemy_pos,
            "food_pos": self.food_pos,
            "player_direction": self.player_direction,
            "score": self.score,
            "enemy_score": self.enemy_score,
            "steps": self.steps,
            "map": self.map
        }

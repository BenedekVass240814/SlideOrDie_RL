"""
Configuration and hyperparameters for RL training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration."""
    width: int = 1600
    height: int = 900
    block_size: int = 50
    max_steps: int = 300  # Shorter episodes for faster training
    chase_mode: bool = False
    
    # Win condition settings
    win_mode: str = "score_compare"  # "first_to_n" or "score_compare"
    win_score: int = 10  # For "first_to_n" mode
    # For "score_compare" mode: episode ends at max_steps, whoever has more points wins
    
    # Opponent settings
    enable_opponent: bool = True
    opponent_delay: int = 5  # Steps between opponent moves
    opponent_type: str = "random"  # "random", "simple", "pathfinding"


@dataclass
class RewardConfig:
    """Reward shaping configuration."""
    food_collected: float = 10.0
    enemy_collected: float = -5.0
    caught_by_enemy: float = -10.0
    
    # Distance-based shaping (key for efficient learning!)
    closer_to_food: float = 0.1
    further_from_food: float = -0.05
    
    # Survival bonus
    time_penalty: float = -0.005
    
    # Win/lose (applied at episode end)
    win_bonus: float = 20.0
    lose_penalty: float = -15.0
    draw_penalty: float = -5.0  # Encourage winning, not drawing
    
    # Wall collision penalty
    wall_collision: float = -0.5


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training settings
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10  # PPO epochs per update
    max_grad_norm: float = 0.5
    
    # Network architecture
    hidden_size: int = 128
    n_layers: int = 2


@dataclass
class TrainConfig:
    """Training configuration."""
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    save_freq: int = 25_000
    n_eval_episodes: int = 10
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 3
    
    # Logging
    log_interval: int = 1
    verbose: int = 1
    
    # Rendering during training
    render_freq: int = 0  # 0 = never render during training


# Default configurations
DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()

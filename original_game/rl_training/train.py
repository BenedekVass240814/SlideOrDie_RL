"""
PPO Training script for Slide or Die.
Supports curriculum learning and efficient training.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from typing import Optional, List
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SlideOrDieEnv
from agent import PPOAgent
from config import (
    EnvConfig, RewardConfig, PPOConfig, TrainConfig,
    DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG, DEFAULT_PPO_CONFIG, DEFAULT_TRAIN_CONFIG
)


class CurriculumManager:
    """
    Manages curriculum learning stages.
    Starts easy and progressively increases difficulty.
    """
    
    def __init__(self, n_stages: int = 3):
        self.n_stages = n_stages
        self.current_stage = 0
        self.stage_thresholds = [0.6, 0.7, 0.8]  # Win rate to advance
        self.recent_wins = []
        self.window_size = 50
    
    def get_env_config(self) -> EnvConfig:
        """Get environment config for current curriculum stage."""
        if self.current_stage == 0:
            # Stage 0: No opponent - just learn to collect food
            return EnvConfig(
                enable_opponent=False,
                max_steps=200,
                win_mode="score_compare",  # End after max_steps
            )
        elif self.current_stage == 1:
            # Stage 1: Random opponent
            return EnvConfig(
                enable_opponent=True,
                opponent_type="random",
                opponent_delay=8,  # Slow opponent
                max_steps=250,
                win_mode="score_compare",
            )
        elif self.current_stage == 2:
            # Stage 2: Simple chasing opponent
            return EnvConfig(
                enable_opponent=True,
                opponent_type="simple",
                opponent_delay=5,
                max_steps=300,
                win_mode="score_compare",
            )
        else:
            # Stage 3+: Full difficulty
            return EnvConfig(
                enable_opponent=True,
                opponent_type="simple",
                opponent_delay=3,
                max_steps=300,
                win_mode="score_compare",
            )
    
    def record_episode(self, won: bool, score: int, enemy_score: int):
        """Record episode result for curriculum advancement."""
        self.recent_wins.append(1 if won else 0)
        if len(self.recent_wins) > self.window_size:
            self.recent_wins.pop(0)
    
    def should_advance(self) -> bool:
        """Check if should advance to next curriculum stage."""
        if len(self.recent_wins) < self.window_size:
            return False
        
        win_rate = sum(self.recent_wins) / len(self.recent_wins)
        threshold = self.stage_thresholds[min(self.current_stage, len(self.stage_thresholds) - 1)]
        
        return win_rate >= threshold
    
    def advance(self):
        """Advance to next curriculum stage."""
        if self.current_stage < self.n_stages:
            self.current_stage += 1
            self.recent_wins = []
            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCED TO STAGE {self.current_stage}")
            print(f"{'='*60}\n")


class Trainer:
    """Main training class."""
    
    def __init__(
        self,
        train_config: Optional[TrainConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        use_curriculum: bool = True,
        save_dir: str = "checkpoints",
    ):
        self.train_config = train_config or DEFAULT_TRAIN_CONFIG
        self.ppo_config = ppo_config or DEFAULT_PPO_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.use_curriculum = use_curriculum
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Curriculum manager
        self.curriculum = CurriculumManager(self.train_config.curriculum_stages) if use_curriculum else None
        
        # Initialize environment
        self._create_env()
        
        # Initialize agent
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = PPOAgent(obs_dim, action_dim, self.ppo_config)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.best_avg_reward = float('-inf')
    
    def _create_env(self, render_mode: str = None):
        """Create or recreate environment (for curriculum)."""
        if self.use_curriculum:
            env_config = self.curriculum.get_env_config()
        else:
            env_config = DEFAULT_ENV_CONFIG
        
        self.env = SlideOrDieEnv(
            render_mode=render_mode,
            env_config=env_config,
            reward_config=self.reward_config
        )
    
    def collect_rollout(self) -> dict:
        """Collect one rollout of experiences."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        rollout_stats = {
            "rewards": [],
            "lengths": [],
            "wins": [],
            "scores": [],
            "enemy_scores": [],
        }
        
        for step in range(self.ppo_config.n_steps):
            # Get action
            action, log_prob, value = self.agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            self.agent.total_steps += 1
            
            if done:
                rollout_stats["rewards"].append(episode_reward)
                rollout_stats["lengths"].append(episode_length)
                rollout_stats["wins"].append(info.get("won", False))
                rollout_stats["scores"].append(self.env.game.score)
                rollout_stats["enemy_scores"].append(self.env.game.enemy_score)
                
                # Curriculum tracking
                if self.curriculum:
                    self.curriculum.record_episode(
                        info.get("won", False),
                        self.env.game.score,
                        self.env.game.enemy_score
                    )
                
                # Reset
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        return rollout_stats, obs
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Total timesteps: {self.train_config.total_timesteps:,}")
        print(f"Curriculum learning: {self.use_curriculum}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        while self.agent.total_steps < self.train_config.total_timesteps:
            # Collect rollout
            rollout_stats, last_obs = self.collect_rollout()
            
            # Update policy
            update_stats = self.agent.update(last_obs)
            
            # Track stats
            if rollout_stats["rewards"]:
                self.episode_rewards.extend(rollout_stats["rewards"])
                self.episode_lengths.extend(rollout_stats["lengths"])
                self.episode_wins.extend(rollout_stats["wins"])
            
            # Curriculum advancement
            if self.curriculum and self.curriculum.should_advance():
                self.curriculum.advance()
                self._create_env()
            
            # Logging
            if len(self.episode_rewards) >= self.train_config.log_interval:
                recent_rewards = self.episode_rewards[-100:]
                recent_wins = self.episode_wins[-100:]
                
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) if recent_wins else 0
                
                elapsed = time.time() - start_time
                fps = self.agent.total_steps / elapsed
                
                stage_str = f" | Stage: {self.curriculum.current_stage}" if self.curriculum else ""
                
                print(f"Steps: {self.agent.total_steps:>8,} | "
                      f"Episodes: {len(self.episode_rewards):>5} | "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"FPS: {fps:.0f}{stage_str}")
            
            # Save checkpoint
            if self.agent.total_steps % self.train_config.save_freq == 0:
                self._save_checkpoint()
            
            # Evaluate
            if self.agent.total_steps % self.train_config.eval_freq == 0:
                self._evaluate()
        
        # Final save
        self._save_checkpoint(final=True)
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
    
    def _evaluate(self):
        """Evaluate current policy."""
        eval_rewards = []
        eval_wins = []
        
        # Create eval environment
        eval_env = SlideOrDieEnv(
            env_config=EnvConfig(enable_opponent=True, opponent_type="simple", opponent_delay=5)
        )
        
        for _ in range(self.train_config.n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            eval_wins.append(info.get("won", False))
        
        eval_env.close()
        
        avg_reward = np.mean(eval_rewards)
        win_rate = np.mean(eval_wins)
        
        print(f"\n--- EVALUATION ---")
        print(f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.1%}")
        print(f"------------------\n")
        
        # Save best model
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.agent.save(os.path.join(self.save_dir, "best_model.pth"))
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if final:
            filename = f"ppo_final_{timestamp}.pth"
        else:
            filename = f"ppo_step{self.agent.total_steps}_{timestamp}.pth"
        
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        
        # Save training stats
        stats_path = os.path.join(self.save_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "episode_rewards": self.episode_rewards[-1000:],
                "episode_wins": self.episode_wins[-1000:],
                "total_steps": self.agent.total_steps,
                "curriculum_stage": self.curriculum.current_stage if self.curriculum else None,
            }, f)


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for Slide or Die")
    parser.add_argument("--timesteps", type=int, default=500000, 
                        help="Total training timesteps")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Create configs
    train_config = TrainConfig(total_timesteps=args.timesteps)
    ppo_config = PPOConfig(learning_rate=args.lr)
    
    # Create trainer
    trainer = Trainer(
        train_config=train_config,
        ppo_config=ppo_config,
        use_curriculum=not args.no_curriculum,
        save_dir=args.save_dir,
    )
    
    # Train!
    trainer.train()


if __name__ == "__main__":
    main()

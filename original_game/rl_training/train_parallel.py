"""
Parallel PPO Training with Vectorized Environments.
Significantly faster than single-environment training!
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from typing import Optional
import json
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ParallelRolloutBuffer:
    """Rollout buffer for vectorized environments."""
    
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: str = "cpu"):
        import torch
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.device = device
        self.ptr = 0
        
        # Pre-allocate buffers
        self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=bool)
    
    def add(self, obs, actions, rewards, values, log_probs, dones):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones
        self.ptr += 1
    
    def reset(self):
        self.ptr = 0
    
    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE advantages across all environments."""
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs)
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        self.advantages = advantages
        self.returns = advantages + self.values
        
        # Flatten and normalize
        self.flat_obs = self.observations.reshape(-1, self.obs_dim)
        self.flat_actions = self.actions.reshape(-1)
        self.flat_log_probs = self.log_probs.reshape(-1)
        self.flat_advantages = self.advantages.reshape(-1)
        self.flat_returns = self.returns.reshape(-1)
        
        # Normalize advantages
        self.flat_advantages = (self.flat_advantages - self.flat_advantages.mean()) / (self.flat_advantages.std() + 1e-8)
    
    def get_batches(self, batch_size: int):
        """Generate random minibatches."""
        import torch
        
        n_samples = self.n_steps * self.n_envs
        indices = np.random.permutation(n_samples)
        
        obs = torch.FloatTensor(self.flat_obs).to(self.device)
        actions = torch.LongTensor(self.flat_actions).to(self.device)
        log_probs = torch.FloatTensor(self.flat_log_probs).to(self.device)
        advantages = torch.FloatTensor(self.flat_advantages).to(self.device)
        returns = torch.FloatTensor(self.flat_returns).to(self.device)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            yield (
                obs[batch_idx],
                actions[batch_idx],
                log_probs[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
            )


class ParallelTrainer:
    """
    Parallel PPO trainer using vectorized environments.
    
    Key benefits:
    - Multiple environments step in parallel
    - GPU processes batched observations efficiently
    - 3-10x faster than single-environment training
    """
    
    def __init__(
        self,
        n_envs: int = 8,
        train_config = None,
        ppo_config = None,
        reward_config = None,
        env_config = None,
        save_dir: str = "checkpoints",
        use_multiprocessing: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "slide-or-die-rl",
        wandb_run_name: str = None,
    ):
        # Import here to avoid multiprocessing pickle issues on Windows
        from env import SlideOrDieEnv
        from agent import PPOAgent
        from vec_env import make_vec_env
        from config import (
            EnvConfig, RewardConfig, PPOConfig, TrainConfig,
            DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG, DEFAULT_PPO_CONFIG, DEFAULT_TRAIN_CONFIG
        )
        
        # Wandb setup
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"ppo_{n_envs}envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "algorithm": "PPO",
                    "n_envs": n_envs,
                    "total_timesteps": train_config.total_timesteps if train_config else 500000,
                    "learning_rate": ppo_config.learning_rate if ppo_config else 3e-4,
                    "gamma": ppo_config.gamma if ppo_config else 0.99,
                    "gae_lambda": ppo_config.gae_lambda if ppo_config else 0.95,
                    "clip_epsilon": ppo_config.clip_epsilon if ppo_config else 0.2,
                    "n_epochs": ppo_config.n_epochs if ppo_config else 10,
                    "batch_size": ppo_config.batch_size if ppo_config else 64,
                    "entropy_coef": ppo_config.entropy_coef if ppo_config else 0.01,
                    "max_steps_per_episode": 300,
                    "win_mode": "score_compare",
                }
            )
        else:
            self.wandb = None
        
        self.n_envs = n_envs
        self.train_config = train_config or DEFAULT_TRAIN_CONFIG
        self.ppo_config = ppo_config or DEFAULT_PPO_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        # Use score_compare mode for faster training episodes
        self.env_config = env_config or EnvConfig(
            enable_opponent=True, 
            opponent_type="simple",
            max_steps=300,
            win_mode="score_compare"
        )
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create vectorized environment
        print(f"Creating {n_envs} parallel environments...")
        self.vec_env = make_vec_env(
            SlideOrDieEnv,
            n_envs=n_envs,
            use_multiprocessing=use_multiprocessing,
            env_config=self.env_config,
            reward_config=self.reward_config,
        )
        
        # Initialize agent
        obs_dim = self.vec_env.observation_space.shape[0]
        action_dim = self.vec_env.action_space.n
        self.agent = PPOAgent(obs_dim, action_dim, self.ppo_config)
        
        # Parallel buffer
        self.buffer = ParallelRolloutBuffer(
            n_steps=self.ppo_config.n_steps // n_envs,  # Steps per env
            n_envs=n_envs,
            obs_dim=obs_dim,
            device=self.agent.device,
        )
        
        # Stats
        self.episode_rewards = []
        self.episode_wins = []
        self.best_avg_reward = float('-inf')
        
        print(f"{'='*60}")
        print(f"PARALLEL TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Environments: {n_envs}")
        print(f"Steps per rollout: {self.buffer.n_steps * n_envs}")
        print(f"Effective batch size: {n_envs} (parallel sampling)")
        print(f"{'='*60}\n")
    
    def collect_rollout(self, obs: np.ndarray):
        """Collect rollout from all environments in parallel."""
        import torch
        
        rollout_episode_rewards = []
        rollout_episode_wins = []
        episode_rewards = np.zeros(self.n_envs)
        
        self.buffer.reset()
        
        for step in range(self.buffer.n_steps):
            # Get actions for all envs at once (batched!)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.agent.device)
                actions, log_probs, values = self.agent.policy.get_action(obs_tensor)
                actions = actions.cpu().numpy()
                log_probs = log_probs.cpu().numpy()
                values = values.cpu().numpy()
            
            # Step all environments
            next_obs, rewards, dones, infos = self.vec_env.step(actions)
            
            # Store transitions
            self.buffer.add(obs, actions, rewards, values, log_probs, dones)
            
            # Track episode stats
            episode_rewards += rewards
            self.agent.total_steps += self.n_envs
            
            for i, done in enumerate(dones):
                if done:
                    rollout_episode_rewards.append(episode_rewards[i])
                    rollout_episode_wins.append(infos[i].get("won", False))
                    episode_rewards[i] = 0
            
            obs = next_obs
        
        # Get last values for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.agent.device)
            _, _, last_values = self.agent.policy.get_action(obs_tensor)
            last_values = last_values.cpu().numpy()
        
        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_values,
            self.ppo_config.gamma,
            self.ppo_config.gae_lambda,
        )
        
        return obs, rollout_episode_rewards, rollout_episode_wins
    
    def update(self):
        """PPO update using collected rollout."""
        import torch
        import torch.nn as nn
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.ppo_config.n_epochs):
            for batch in self.buffer.get_batches(self.ppo_config.batch_size):
                obs, actions, old_log_probs, advantages, returns = batch
                
                # Get current policy outputs
                new_log_probs, values, entropy = self.agent.policy.evaluate_actions(obs, actions)
                
                # Policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_config.clip_epsilon,
                                    1 + self.ppo_config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss +
                        self.ppo_config.value_coef * value_loss +
                        self.ppo_config.entropy_coef * entropy_loss)
                
                # Optimize
                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.ppo_config.max_grad_norm)
                self.agent.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                n_updates += 1
        
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def train(self):
        """Main parallel training loop."""
        print("\n" + "="*60)
        print("STARTING PARALLEL TRAINING")
        print("="*60)
        print(f"Total timesteps: {self.train_config.total_timesteps:,}")
        print(f"Parallel environments: {self.n_envs}")
        print("="*60 + "\n")
        
        start_time = time.time()
        obs, _ = self.vec_env.reset(seed=42)
        
        while self.agent.total_steps < self.train_config.total_timesteps:
            # Collect rollout (parallel!)
            obs, ep_rewards, ep_wins = self.collect_rollout(obs)
            
            # Update policy
            update_stats = self.update()
            
            # Track stats
            self.episode_rewards.extend(ep_rewards)
            self.episode_wins.extend(ep_wins)
            
            # Logging
            if ep_rewards:
                recent_rewards = self.episode_rewards[-100:]
                recent_wins = self.episode_wins[-100:]
                
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) if recent_wins else 0
                
                elapsed = time.time() - start_time
                fps = self.agent.total_steps / elapsed
                
                print(f"Steps: {self.agent.total_steps:>8,} | "
                      f"Episodes: {len(self.episode_rewards):>5} | "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"FPS: {fps:.0f}")
                
                # Log to wandb
                if self.use_wandb:
                    self.wandb.log({
                        "train/avg_reward": avg_reward,
                        "train/win_rate": win_rate,
                        "train/episodes": len(self.episode_rewards),
                        "train/fps": fps,
                        "train/policy_loss": update_stats["policy_loss"],
                        "train/value_loss": update_stats["value_loss"],
                        "train/entropy": update_stats["entropy"],
                        "train/total_loss": update_stats["loss"],
                    }, step=self.agent.total_steps)
            
            # Save checkpoint
            if self.agent.total_steps % self.train_config.save_freq == 0:
                self._save_checkpoint()
        
        # Final save
        self._save_checkpoint(final=True)
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Average FPS: {self.agent.total_steps/elapsed:.0f}")
        print("="*60)
        
        # Close wandb
        if self.use_wandb:
            # Log final metrics
            self.wandb.log({
                "final/total_episodes": len(self.episode_rewards),
                "final/best_avg_reward": self.best_avg_reward,
                "final/training_time_minutes": elapsed / 60,
                "final/avg_fps": self.agent.total_steps / elapsed,
            })
            self.wandb.finish()
        
        self.vec_env.close()
    
    def _save_checkpoint(self, final: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if final:
            filename = f"ppo_parallel_final_{timestamp}.pth"
        else:
            filename = f"ppo_parallel_step{self.agent.total_steps}_{timestamp}.pth"
        
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        
        # Also save as best if applicable
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-100:])
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.agent.save(os.path.join(self.save_dir, "best_model.pth"))


def main():
    import argparse
    from config import TrainConfig, PPOConfig
    
    parser = argparse.ArgumentParser(description="Parallel PPO training for Slide or Die")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--no-mp", action="store_true",
                        help="Disable multiprocessing (for debugging)")
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="slide-or-die-rl",
                        help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Wandb run name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    train_config = TrainConfig(total_timesteps=args.timesteps)
    ppo_config = PPOConfig(learning_rate=args.lr)
    
    trainer = ParallelTrainer(
        n_envs=args.n_envs,
        train_config=train_config,
        ppo_config=ppo_config,
        save_dir=args.save_dir,
        use_multiprocessing=not args.no_mp,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
    )
    
    trainer.train()


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()

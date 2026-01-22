"""
Improved Parallel Training Script V2 with:
1. Frame stacking (temporal memory)
2. Larger network
3. Better reward shaping
4. Wandb integration (centralized config)
5. Entropy annealing
6. Fixed map training with periodic regeneration
"""

import os
import sys
import time
import multiprocessing as mp
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import SlideOrDieEnvV2, EnvConfig
from agent_v2 import PPOAgentV2, ActorCriticV2
from wandb_config import (
    WandbConfig, MapConfig, 
    DEFAULT_WANDB_CONFIG, DEFAULT_MAP_CONFIG,
    init_wandb, log_metrics
)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed - training will continue without remote logging")


@dataclass
class TrainingConfigV2:
    """Training configuration."""
    # Environment
    n_envs: int = 8
    frame_stack: int = 4
    max_steps: int = 300
    
    # Training
    total_timesteps: int = 10_000_000
    n_steps: int = 256  # Steps per environment per rollout
    n_epochs: int = 4
    batch_size: int = 256
    
    # PPO
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02  # Start higher for exploration
    entropy_min: float = 0.001  # Anneal to this
    max_grad_norm: float = 0.5
    
    # Network
    hidden_size: int = 256
    n_layers: int = 3
    use_lstm: bool = False
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Wandb (use centralized config)
    wandb_config: WandbConfig = field(default_factory=lambda: DEFAULT_WANDB_CONFIG)
    
    # Map settings (use centralized config)
    map_config: MapConfig = field(default_factory=lambda: DEFAULT_MAP_CONFIG)


class EnvFactory:
    """Picklable environment factory for Windows multiprocessing."""
    
    def __init__(self, env_config: EnvConfig, map_config: MapConfig, frame_stack: int = 4):
        self.env_config = env_config
        self.map_config = map_config
        self.frame_stack = frame_stack
    
    def __call__(self):
        return SlideOrDieEnvV2(
            env_config=self.env_config, 
            map_config=self.map_config,
            frame_stack=self.frame_stack
        )


def worker_process(
    conn: mp.connection.Connection,
    env_fn,
):
    """Worker process for running environments."""
    env = env_fn()
    episode_reward = 0.0
    
    while True:
        cmd, data = conn.recv()
        
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            episode_reward += reward
            done = terminated or truncated
            if done:
                # Auto-reset and return new observation with episode stats
                final_info = info.copy()
                final_info["episode_reward"] = episode_reward
                episode_reward = 0.0
                obs, _ = env.reset()
                conn.send((obs, reward, done, final_info))
            else:
                conn.send((obs, reward, done, info))
        
        elif cmd == "reset":
            episode_reward = 0.0
            obs, info = env.reset()
            conn.send((obs, info))
        
        elif cmd == "close":
            env.close()
            conn.close()
            break


class VecEnvV2:
    """Vectorized environment using multiprocessing."""
    
    def __init__(self, env_fns: List, start_method: str = "spawn"):
        self.n_envs = len(env_fns)
        self.waiting = False
        
        # Create environment to get observation/action space
        temp_env = env_fns[0]()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()
        
        # Create workers
        ctx = mp.get_context(start_method)
        self.parent_conns = []
        self.child_conns = []
        self.processes = []
        
        for i, env_fn in enumerate(env_fns):
            parent_conn, child_conn = ctx.Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            
            process = ctx.Process(
                target=worker_process,
                args=(child_conn, env_fn),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset all environments."""
        for conn in self.parent_conns:
            conn.send(("reset", None))
        
        obs_list = []
        infos = []
        for conn in self.parent_conns:
            obs, info = conn.recv()
            obs_list.append(obs)
            infos.append(info)
        
        return np.stack(obs_list), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))
        
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for conn in self.parent_conns:
            obs, reward, done, info = conn.recv()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )
    
    def close(self):
        """Close all environments."""
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except:
                pass
        
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, n_envs: int, n_steps: int, obs_dim: int, device: str):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.device = device
        
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=bool)
        
        self.pos = 0
    
    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray,
    ):
        """Add a step to the buffer."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs
        self.dones[self.pos] = dones
        self.pos += 1
    
    def reset(self):
        """Reset the buffer."""
        self.pos = 0
    
    def compute_returns(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages using GAE."""
        advantages = np.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + self.values
        return returns, advantages
    
    def get_batches(
        self,
        batch_size: int,
        returns: np.ndarray,
        advantages: np.ndarray,
    ):
        """Generate minibatches for training."""
        total_samples = self.n_steps * self.n_envs
        indices = np.random.permutation(total_samples)
        
        # Flatten all data
        flat_obs = self.obs.reshape(total_samples, -1)
        flat_actions = self.actions.reshape(total_samples)
        flat_log_probs = self.log_probs.reshape(total_samples)
        flat_values = self.values.reshape(total_samples)
        flat_returns = returns.reshape(total_samples)
        flat_advantages = advantages.reshape(total_samples)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # Generate batches
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(flat_obs[batch_indices]).to(self.device),
                torch.LongTensor(flat_actions[batch_indices]).to(self.device),
                torch.FloatTensor(flat_log_probs[batch_indices]).to(self.device),
                torch.FloatTensor(flat_values[batch_indices]).to(self.device),
                torch.FloatTensor(flat_returns[batch_indices]).to(self.device),
                torch.FloatTensor(flat_advantages[batch_indices]).to(self.device),
            )


class ParallelTrainerV2:
    """Improved parallel trainer."""
    
    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create environment config
        self.env_config = EnvConfig(
            max_steps=config.max_steps,
            win_mode="score_compare",
        )
        
        # Create vectorized environment using picklable factory class
        print(f"Creating {config.n_envs} parallel environments...")
        print(f"Map settings: fixed={config.map_config.use_fixed_map}, change_every={config.map_config.map_change_episodes} episodes")
        env_fns = [
            EnvFactory(self.env_config, config.map_config, config.frame_stack)
            for _ in range(config.n_envs)
        ]
        self.vec_env = VecEnvV2(env_fns, start_method="spawn")
        
        # Get observation and action dimensions
        self.obs_dim = self.vec_env.observation_space.shape[0]
        self.action_dim = self.vec_env.action_space.n
        
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.action_dim}")
        
        # Create agent
        self.agent = PPOAgentV2(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            learning_rate=config.learning_rate,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            use_lstm=config.use_lstm,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            device=self.device,
        )
        
        # Create rollout buffer
        self.buffer = RolloutBuffer(
            n_envs=config.n_envs,
            n_steps=config.n_steps,
            obs_dim=self.obs_dim,
            device=self.device,
        )
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        self.total_steps = 0
        self.n_updates = 0
        
        # Current entropy coefficient (will be annealed)
        self.current_entropy_coef = config.entropy_coef
        
        # Checkpoints directory
        self.checkpoint_dir = os.path.join(
            os.path.dirname(__file__),
            "checkpoints_v2",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb using centralized config
        wandb_cfg = config.wandb_config
        self.use_wandb = wandb_cfg.enabled and WANDB_AVAILABLE
        self.wandb_run = None
        
        if self.use_wandb:
            training_params = {
                "version": 2,
                "n_envs": config.n_envs,
                "frame_stack": config.frame_stack,
                "hidden_size": config.hidden_size,
                "map_change_episodes": config.map_config.map_change_episodes,
                "n_steps": config.n_steps,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "n_layers": config.n_layers,
                "use_lstm": config.use_lstm,
                "gamma": config.gamma,
                "entropy_coef": config.entropy_coef,
                "clip_epsilon": config.clip_epsilon,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "use_fixed_map": config.map_config.use_fixed_map,
                "max_steps": config.max_steps,
            }
            self.wandb_run = init_wandb(wandb_cfg, training_params)
    
    def collect_rollouts(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect rollouts from all environments."""
        self.buffer.reset()
        
        for step in range(self.config.n_steps):
            # Get actions from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action_logits, values, _ = self.agent.policy(obs_tensor)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            actions_np = actions.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            
            # Step environments
            next_obs, rewards, dones, infos = self.vec_env.step(actions_np)
            
            # Store in buffer
            self.buffer.add(
                obs=obs,
                actions=actions_np,
                rewards=rewards,
                values=values_np,
                log_probs=log_probs_np,
                dones=dones,
            )
            
            # Track episode stats
            for i, info in enumerate(infos):
                if dones[i]:
                    self.episode_rewards.append(info.get("episode_reward", 0))
                    self.episode_lengths.append(info.get("steps", 0))
                    self.win_rates.append(1.0 if info.get("won", False) else 0.0)
            
            obs = next_obs
            self.total_steps += self.config.n_envs
        
        # Compute last values for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            _, last_values, _ = self.agent.policy(obs_tensor)
            last_values = last_values.squeeze(-1).cpu().numpy()
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns(
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        return obs, returns, advantages
    
    def train_epoch(self, returns: np.ndarray, advantages: np.ndarray) -> dict:
        """Train for one epoch on collected data."""
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        for batch in self.buffer.get_batches(self.config.batch_size, returns, advantages):
            obs, actions, old_log_probs, old_values, batch_returns, batch_advantages = batch
            
            # Get current policy outputs
            log_probs, values, entropy = self.agent.policy.evaluate_actions(obs, actions)
            
            # Policy loss (PPO clipping)
            ratio = torch.exp(log_probs - old_log_probs)
            clip_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
            policy_loss = -torch.min(ratio * batch_advantages, clip_ratio * batch_advantages).mean()
            
            # Value loss (clipped)
            value_clipped = old_values + torch.clamp(
                values - old_values, -self.config.clip_epsilon, self.config.clip_epsilon
            )
            value_loss_unclipped = (values - batch_returns) ** 2
            value_loss_clipped = (value_clipped - batch_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Entropy loss (use current annealed coefficient)
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.current_entropy_coef * entropy_loss
            
            # Optimize
            self.agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.config.max_grad_norm)
            self.agent.optimizer.step()
            
            # Track metrics
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            
            with torch.no_grad():
                clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().item()
                clip_fractions.append(clip_fraction)
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
        }
    
    def anneal_entropy(self):
        """Anneal entropy coefficient over training."""
        progress = min(1.0, self.total_steps / self.config.total_timesteps)
        self.current_entropy_coef = self.config.entropy_coef + (
            self.config.entropy_min - self.config.entropy_coef
        ) * progress
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training V2")
        print("="*60)
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Environments: {self.config.n_envs}")
        print(f"Steps per rollout: {self.config.n_steps}")
        print(f"Samples per update: {self.config.n_steps * self.config.n_envs:,}")
        print("="*60 + "\n")
        
        start_time = time.time()
        n_rollouts = 0
        
        # Initialize observations ONCE at start (not every rollout!)
        obs, _ = self.vec_env.reset()
        
        try:
            while self.total_steps < self.config.total_timesteps:
                # Collect rollouts - pass obs to maintain continuity
                obs, returns, advantages = self.collect_rollouts(obs)
                n_rollouts += 1
                
                # Train for multiple epochs
                train_metrics = None
                for _ in range(self.config.n_epochs):
                    train_metrics = self.train_epoch(returns, advantages)
                
                self.n_updates += 1
                
                # Anneal entropy
                self.anneal_entropy()
                
                # Update learning rate
                self.agent.scheduler.step()
                
                # Logging
                if n_rollouts % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = self.total_steps / elapsed
                    
                    avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                    avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                    win_rate = np.mean(self.win_rates) if self.win_rates else 0
                    
                    print(f"\n[Step {self.total_steps:,}] ({self.total_steps/self.config.total_timesteps*100:.1f}%)")
                    print(f"  Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate*100:.1f}%")
                    print(f"  Avg Length: {avg_length:.0f} | FPS: {fps:.0f}")
                    print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
                    print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
                    print(f"  Entropy: {train_metrics['entropy']:.4f} (coef: {self.current_entropy_coef:.4f})")
                    print(f"  Clip Fraction: {train_metrics['clip_fraction']:.2%}")
                    print(f"  LR: {self.agent.optimizer.param_groups[0]['lr']:.2e}")
                    
                    if self.use_wandb:
                        metrics = {
                            "train/avg_reward": avg_reward,
                            "train/win_rate": win_rate,
                            "train/episodes": len(self.episode_rewards),
                            "train/fps": fps,
                            "train/policy_loss": train_metrics["policy_loss"],
                            "train/value_loss": train_metrics["value_loss"],
                            "train/entropy": train_metrics["entropy"],
                            "train/total_loss": train_metrics["policy_loss"] + self.config.value_coef * train_metrics["value_loss"],
                        }
                        log_metrics(metrics, self.total_steps, self.wandb_run)
                
                # Save checkpoint
                if n_rollouts % self.config.save_interval == 0:
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        
        finally:
            # Final save
            self.save_checkpoint(final=True)
            self.vec_env.close()
            
            if self.use_wandb and self.wandb_run is not None:
                self.wandb_run.finish()
            
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print("Training Complete!")
            print(f"Total steps: {self.total_steps:,}")
            print(f"Total time: {total_time/3600:.2f} hours")
            print(f"Average FPS: {self.total_steps/total_time:.0f}")
            print(f"{'='*60}")
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if final:
            filename = f"ppo_v2_final_{self.total_steps}_{timestamp}.pth"
        else:
            filename = f"ppo_v2_{self.total_steps}_{timestamp}.pth"
        
        path = os.path.join(self.checkpoint_dir, filename)
        self.agent.save(path)


def main():
    """Main entry point."""
    # Check for multiprocessing on Windows
    mp.freeze_support()
    
    # Create configs
    wandb_cfg = WandbConfig(
        project="slide-or-die-rl",
        run_name_prefix="ppo_v2",
        tags=["ppo", "v2", "frame-stack", "fixed-map"],
        enabled=True,
    )
    
    map_cfg = MapConfig(
        use_fixed_map=True,
        map_change_episodes=500,  # Generate new map every 500 episodes
    )
    
    config = TrainingConfigV2(
        n_envs=8,
        frame_stack=2,  # Reduced from 4 - less complexity, still has velocity info
        max_steps=300,
        total_timesteps=10_000_000,
        n_steps=256,
        n_epochs=4,
        batch_size=256,
        learning_rate=3e-4,  # Slightly higher LR
        hidden_size=128,  # Smaller network - less prone to overfitting
        n_layers=2,  # Fewer layers
        use_lstm=False,
        entropy_coef=0.01,  # Standard entropy
        entropy_min=0.005,
        wandb_config=wandb_cfg,
        map_config=map_cfg,
    )
    
    trainer = ParallelTrainerV2(config)
    trainer.train()


if __name__ == "__main__":
    main()

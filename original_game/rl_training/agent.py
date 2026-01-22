"""
PPO Agent implementation for Slide or Die.
Uses a simple MLP network - no CNN needed with compact observations!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional
from config import PPOConfig, DEFAULT_PPO_CONFIG


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Simple MLP architecture - works great with feature-based observations!
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        
        self.config = config
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_size),
                nn.ReLU(),
            ])
            in_dim = config.hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Returns:
            action, log_prob, value
        """
        action_logits, value = self.forward(x)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs, values, entropy
        """
        action_logits, value = self.forward(x)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer to store rollout experiences for PPO."""
    
    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.ptr = 0
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1
    
    def compute_returns_and_advantages(
        self, 
        last_value: float, 
        gamma: float, 
        gae_lambda: float
    ):
        """Compute GAE advantages and returns."""
        advantages = []
        last_gae = 0
        
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                last_gae = delta
            else:
                delta = self.rewards[t] + gamma * values[t + 1] - values[t]
                last_gae = delta + gamma * gae_lambda * last_gae
            
            advantages.insert(0, last_gae)
        
        self.advantages = np.array(advantages, dtype=np.float32)
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_batches(self, batch_size: int):
        """Generate random minibatches for training."""
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                obs[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
            )


class PPOAgent:
    """PPO Agent for training."""
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        config: Optional[PPOConfig] = None,
        device: str = None
    ):
        self.config = config or DEFAULT_PPO_CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(obs_dim, action_dim, self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.config.learning_rate
        )
        
        self.buffer = RolloutBuffer(
            self.config.n_steps, 
            obs_dim, 
            self.device
        )
        
        # Stats
        self.total_steps = 0
        
        print(f"{'='*60}")
        print("PPO Agent Initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Observation dim: {obs_dim}")
        print(f"Action dim: {action_dim}")
        print(f"Network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"{'='*60}")
    
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.buffer.add(obs, action, reward, value, log_prob, done)
    
    def update(self, last_obs: np.ndarray) -> dict:
        """Perform PPO update using collected rollout."""
        # Get last value for GAE computation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(obs_tensor)
            last_value = last_value.item()
        
        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda
        )
        
        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs, actions, old_log_probs, advantages, returns = batch
                
                # Get current policy outputs
                new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                    1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                        self.config.value_coef * value_loss + 
                        self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                n_updates += 1
        
        # Clear buffer
        self.buffer.reset()
        
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        print(f"Model loaded from {path}")

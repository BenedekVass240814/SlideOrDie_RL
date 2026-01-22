"""
Improved PPO Agent with:
1. Larger network capacity
2. Optional LSTM for sequence modeling
3. Better initialization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional


class ActorCriticV2(nn.Module):
    """
    Larger Actor-Critic network with optional LSTM.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_size: int = 256,
        n_layers: int = 3,
        use_lstm: bool = False,
        lstm_hidden_size: int = 128,
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        
        # Larger feature extractor
        layers = []
        in_dim = obs_dim
        for i in range(n_layers):
            out_dim = hidden_size if i < n_layers - 1 else hidden_size
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),  # LayerNorm for stability
                nn.ReLU(),
                nn.Dropout(0.1) if i < n_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Optional LSTM layer
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            policy_input_size = lstm_hidden_size
        else:
            self.lstm = None
            policy_input_size = hidden_size
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(policy_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(policy_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Better initialization
        self.apply(self._init_weights)
        
        # Initialize final layers with smaller weights
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass."""
        features = self.shared(x)
        
        if self.use_lstm and self.lstm is not None:
            # Add sequence dimension if needed
            if features.dim() == 2:
                features = features.unsqueeze(1)
            
            if lstm_states is None:
                lstm_out, lstm_states = self.lstm(features)
            else:
                lstm_out, lstm_states = self.lstm(features, lstm_states)
            
            features = lstm_out.squeeze(1)
        
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value, lstm_states if self.use_lstm else None
    
    def get_action(
        self, 
        x: torch.Tensor, 
        deterministic: bool = False,
        lstm_states: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Get action from policy."""
        action_logits, value, new_lstm_states = self.forward(x, lstm_states)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1), new_lstm_states
    
    def evaluate_actions(
        self, 
        x: torch.Tensor, 
        actions: torch.Tensor,
        lstm_states: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_logits, value, _ = self.forward(x, lstm_states)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value.squeeze(-1), entropy
    
    def get_initial_lstm_states(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden states."""
        if not self.use_lstm:
            return None
        
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)


class PPOAgentV2:
    """Improved PPO Agent with larger network."""
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int,
        learning_rate: float = 2.5e-4,  # Slightly lower LR for stability
        hidden_size: int = 256,
        n_layers: int = 3,
        use_lstm: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lstm = use_lstm
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.policy = ActorCriticV2(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_lstm=use_lstm,
        ).to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.policy.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=1000,  # Will decay over 1000 updates
        )
        
        self.total_steps = 0
        
        # LSTM states for sequential processing
        self.lstm_states = None
        
        total_params = sum(p.numel() for p in self.policy.parameters())
        
        print(f"{'='*60}")
        print("PPO Agent V2 Initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Network: {n_layers} layers, {hidden_size} hidden size")
        print(f"LSTM: {'Enabled' if use_lstm else 'Disabled'}")
        print(f"Total parameters: {total_params:,}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*60}")
    
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value, self.lstm_states = self.policy.get_action(
                obs_tensor, deterministic, self.lstm_states
            )
        
        return action.item(), log_prob.item(), value.item()
    
    def reset_lstm_states(self, batch_size: int = 1):
        """Reset LSTM states for new episode."""
        if self.use_lstm:
            self.lstm_states = self.policy.get_initial_lstm_states(batch_size, self.device)
        else:
            self.lstm_states = None
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test the network
    obs_dim = 18 * 4  # 18 features * 4 frames
    action_dim = 4
    
    agent = PPOAgentV2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=256,
        n_layers=3,
        use_lstm=False,
    )
    
    # Test forward pass
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(obs)
    print(f"Action: {action}, Log prob: {log_prob:.3f}, Value: {value:.3f}")

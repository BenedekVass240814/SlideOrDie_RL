import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from rl_env import SlideOrDieEnv
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt


# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 1000
MAX_STEPS = 1000


class DQN(nn.Module):
    """Lightweight Deep Q-Network with CNN architecture."""
    
    def __init__(self, input_channels, grid_height, grid_width, n_actions):
        super(DQN, self).__init__()
        
        # CNN with pooling to reduce dimensions
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate size after conv layers
        conv_out_height = grid_height // 4
        conv_out_width = grid_width // 4
        conv_out_size = conv_out_height * conv_out_width * 32
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleDQN(nn.Module):
    """Simple MLP-based DQN (alternative)."""
    
    def __init__(self, input_channels, grid_height, grid_width, n_actions):
        super(SimpleDQN, self).__init__()
        
        input_size = input_channels * grid_height * grid_width
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """Experience replay buffer."""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """DQN Agent for training."""
    
    def __init__(self, env, use_simple=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        self.n_actions = env.action_space.n
        
        # Get observation dimensions
        obs_shape = env.observation_space.shape
        self.input_channels = obs_shape[0]
        self.grid_height = obs_shape[1]
        self.grid_width = obs_shape[2]
        
        # Choose network architecture
        NetworkClass = SimpleDQN if use_simple else DQN
        
        # Initialize networks
        self.policy_net = NetworkClass(self.input_channels, self.grid_height, 
                                       self.grid_width, self.n_actions).to(device)
        self.target_net = NetworkClass(self.input_channels, self.grid_height, 
                                       self.grid_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Training parameters
        self.epsilon = EPSILON_START
        self.steps_done = 0
        
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        
        print(f"{'='*80}")
        print(f"DQN AGENT INITIALIZED")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print(f"Network type: {'Simple MLP' if use_simple else 'CNN with pooling'}")
        print(f"Total parameters: {total_params:,}")
        print(f"Action space: {self.n_actions} (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)")
        print(f"Observation space: {obs_shape}")
        print(f"{'='*80}\n")
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def optimize_model(self):
        """Perform one optimization step."""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            next_state_values = next_state_values * (1 - done_batch)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)
        
        # Compute loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=EPISODES, save_freq=100, log_freq=10):
        """Train the agent."""
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training statistics
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'player_scores': [],
            'enemy_scores': [],
            'win_rate': [],
            'epsilon': [],
            'loss': []
        }
        
        win_rate_tracker = deque(maxlen=100)
        
        print(f"{'='*80}")
        print(f"TRAINING START")
        print(f"{'='*80}")
        print(f"Episodes: {episodes}")
        print(f"Max steps per episode: {MAX_STEPS}")
        print(f"Bot difficulty: randomness={self.env.bot_randomness}, delay={self.env.delay}")
        print(f"Save frequency: every {save_freq} episodes")
        print(f"{'='*80}\n")
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(MAX_STEPS):
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.memory.push(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Optimize
                loss = self.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)
                
                if done:
                    break
            
            # Update statistics
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(step + 1)
            stats['player_scores'].append(info['player_score'])
            stats['enemy_scores'].append(info['enemy_score'])
            stats['epsilon'].append(self.epsilon)
            
            if episode_losses:
                stats['loss'].append(np.mean(episode_losses))
            else:
                stats['loss'].append(0)
            
            # Update win rate
            win_rate_tracker.append(1 if info['player_score'] >= 10 else 0)
            current_win_rate = np.mean(win_rate_tracker)
            stats['win_rate'].append(current_win_rate)
            
            # Decay epsilon
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
            
            # Update target network
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Logging
            if episode % log_freq == 0:
                recent_episodes = min(log_freq, len(stats['episode_rewards']))
                avg_reward = np.mean(stats['episode_rewards'][-recent_episodes:])
                avg_length = np.mean(stats['episode_lengths'][-recent_episodes:])
                avg_loss = np.mean(stats['loss'][-recent_episodes:])
                avg_player = np.mean(stats['player_scores'][-recent_episodes:])
                avg_enemy = np.mean(stats['enemy_scores'][-recent_episodes:])
                
                print(f"Ep {episode:4d} | "
                      f"R: {avg_reward:7.2f} | "
                      f"L: {avg_length:5.1f} | "
                      f"Loss: {avg_loss:6.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Win: {current_win_rate:5.1%} | "
                      f"Score: {avg_player:.1f}-{avg_enemy:.1f}")
            
            # Save checkpoint
            if episode % save_freq == 0 and episode > 0:
                checkpoint_path = f'checkpoints/dqn_ep{episode}_{timestamp}.pth'
                self.save_checkpoint(checkpoint_path, episode, stats)
                print(f"→ Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = f'checkpoints/dqn_final_{timestamp}.pth'
        self.save_checkpoint(final_path, episodes, stats)
        
        # Save training stats
        stats_path = f'logs/training_stats_{timestamp}.json'
        with open(stats_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in stats.items()}, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final model: {final_path}")
        print(f"Training stats: {stats_path}")
        print(f"Final win rate (last 100): {np.mean(win_rate_tracker):.1%}")
        print(f"{'='*80}\n")
        
        return stats
    
    def save_checkpoint(self, path, episode, stats):
        """Save checkpoint."""
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': stats
        }, path)
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        print(f"✓ Loaded checkpoint from episode {checkpoint['episode']}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        
        return checkpoint.get('stats', None)
    
    def evaluate(self, n_episodes=10, render=False):
        """Evaluate the trained agent."""
        print(f"\n{'='*80}")
        print(f"EVALUATION ({n_episodes} episodes)")
        print(f"{'='*80}\n")
        
        if render:
            eval_env = SlideOrDieEnv(render_mode='human', bot_randomness=0.3, 
                                     delay=3, speed=15)
        else:
            eval_env = self.env
        
        results = {
            'rewards': [],
            'player_scores': [],
            'enemy_scores': [],
            'steps': [],
            'wins': 0
        }
        
        for episode in range(n_episodes):
            if render:
                state, _ = eval_env.reset()
            else:
                state, _ = self.env.reset()
            
            episode_reward = 0
            
            for step in range(MAX_STEPS):
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = (eval_env if render else self.env).step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            results['rewards'].append(episode_reward)
            results['player_scores'].append(info['player_score'])
            results['enemy_scores'].append(info['enemy_score'])
            results['steps'].append(step + 1)
            
            if info['player_score'] >= 10:
                results['wins'] += 1
                result_str = "WIN  ✓"
            else:
                result_str = "LOSS ✗"
            
            print(f"Ep {episode+1:2d}: {result_str} | "
                  f"Reward: {episode_reward:6.2f} | "
                  f"Score: {info['player_score']:2d}-{info['enemy_score']:2d} | "
                  f"Steps: {step+1:4d}")
        
        if render:
            eval_env.close()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Average reward:       {np.mean(results['rewards']):7.2f}")
        print(f"Average player score: {np.mean(results['player_scores']):7.2f}")
        print(f"Average enemy score:  {np.mean(results['enemy_scores']):7.2f}")
        print(f"Average steps:        {np.mean(results['steps']):7.1f}")
        print(f"Win rate:             {results['wins']/n_episodes:7.1%} ({results['wins']}/{n_episodes})")
        print(f"{'='*80}\n")
        
        return results


def plot_training_stats(stats, save_path='logs/training_plot.png'):
    """Plot training statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Statistics', fontsize=16)
    
    # Plot 1: Rewards
    axes[0, 0].plot(stats['episode_rewards'], alpha=0.3, label='Raw')
    window = 50
    if len(stats['episode_rewards']) >= window:
        smoothed = np.convolve(stats['episode_rewards'], 
                              np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(stats['episode_rewards'])), 
                       smoothed, label=f'Smoothed ({window})')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Win Rate
    axes[0, 1].plot(stats['win_rate'])
    axes[0, 1].set_title('Win Rate (100-episode window)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[0, 2].plot(stats['player_scores'], label='Player', alpha=0.6)
    axes[0, 2].plot(stats['enemy_scores'], label='Enemy', alpha=0.6)
    axes[0, 2].set_title('Scores per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Episode Length
    axes[1, 0].plot(stats['episode_lengths'], alpha=0.5)
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Loss
    axes[1, 1].plot(stats['loss'], alpha=0.5)
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Epsilon
    axes[1, 2].plot(stats['epsilon'])
    axes[1, 2].set_title('Exploration Rate (Epsilon)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved: {save_path}")
    plt.close()


def main():
    """Main training function."""
    # Create environment
    env = SlideOrDieEnv(
        render_mode=None,
        bot_randomness=0.3,
        speed=20,
        chase=False,
        delay=4,
        max_steps=MAX_STEPS
    )
    
    # Create agent (use_simple=False for CNN, True for MLP)
    agent = DQNAgent(env, use_simple=False)
    
    # Train
    stats = agent.train(episodes=EPISODES, save_freq=100, log_freq=10)
    
    # Plot results
    plot_training_stats(stats)
    
    # Evaluate
    print("\nEvaluating trained agent (headless)...")
    agent.evaluate(n_episodes=20, render=False)
    
    print("\nEvaluating with rendering (5 episodes)...")
    agent.evaluate(n_episodes=5, render=True)
    
    env.close()


if __name__ == "__main__":
    main()
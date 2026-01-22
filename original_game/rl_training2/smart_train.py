import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from smart_rl_env import SmartSlideOrDieEnv
import os

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128          # Larger batch for more stable updates
GAMMA = 0.99              # Discount factor
EPS_START = 1.0           # Exploration start
EPS_END = 0.05            # Exploration end (keep some randomness)
EPS_DECAY = 0.999         # Slower decay to explore more
LR = 0.0005               # Learning rate
MEMORY_SIZE = 50000       # Larger memory
TARGET_UPDATE = 20        # Update target net every X episodes

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture. 
    Splits Value (V) and Advantage (A) streams.
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value Stream (Estimates state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (Estimates action advantage A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class Agent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps = 0
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
            
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Double DQN Logic
        # 1. Select best action using Policy Net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # 2. Evaluate that action using Target Net
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    env = SmartSlideOrDieEnv(render_mode=None, speed=1000) # Fast training
    agent = Agent(state_dim=12, action_dim=4)
    
    episodes = 2000
    epsilon = EPS_START
    
    print("Starting Training on Smart Environment...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
        # Decay Epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f} | Score: {env.game.score}")

    print("Training Complete. Saving model...")
    torch.save(agent.policy_net.state_dict(), "smart_dqn.pth")
    
    # --- WATCH IT PLAY ---
    print("Evaluating...")
    env = SmartSlideOrDieEnv(render_mode='human', speed=30)
    agent.policy_net.eval()
    
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            # Greedy action
            action = agent.select_action(state, epsilon=0) 
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
    env.close()

if __name__ == "__main__":
    main()
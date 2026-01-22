import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
import torch
# Assuming your environment is in smart_rl_env.py or rl_env.py
from smart_rl_env import SmartSlideOrDieEnv 

def make_env(render_mode=None):
    """Helper function to create an environment instance."""
    return lambda: SmartSlideOrDieEnv(render_mode=render_mode)

def train_parallel():
    # 1. Setup Parallel Environments (e.g., 32 parallel games)
    num_envs = 32 
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    # 2. Setup Evaluation Env with Video Recording
    # Records an episode every time we call eval_env.reset()
    eval_env = RecordVideo(
        SmartSlideOrDieEnv(render_mode='rgb_array'), 
        video_folder="./best_videos",
        episode_trigger=lambda x: True # Record every eval session
    )

    # 3. Update DQNAgent to handle batches
    # The select_action and optimize_model methods in DQNAgent 
    # must be updated to handle (num_envs, observation_dim)
    
    states, _ = envs.reset()
    best_reward = -float('inf')

    for episode in range(TOTAL_EPISODES):
        # Parallel Step
        actions = agent.select_action_vector(states) # Select action for ALL envs
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Store all transitions in memory
        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i], terminateds[i])
        
        states = next_states
        
        # Periodic Evaluation and Video Saving
        if episode % 100 == 0:
            avg_eval_reward = evaluate_model(agent, eval_env)
            if avg_eval_reward > best_reward:
                best_reward = avg_eval_reward
                # The RecordVideo wrapper saves the file automatically here
                print(f"New best model! Video saved to ./best_videos")
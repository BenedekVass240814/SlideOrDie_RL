"""
Script to play/visualize the trained agent.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SlideOrDieEnv
from agent import PPOAgent
from config import EnvConfig, DEFAULT_PPO_CONFIG


def play_episode(env, agent, delay: float = 0.05):
    """Play a single episode with rendering."""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    terminated = False
    
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        env.render()
        time.sleep(delay)
    
    # Get final scores
    player_score = env.game.score
    enemy_score = env.game.enemy_score
    
    # Determine result based on actual scores
    if terminated:
        # Game ended by reaching win_score
        won = info.get("won", player_score > enemy_score)
    else:
        # Truncated (max steps) - compare scores
        won = player_score > enemy_score
        info["truncated"] = True
    
    info["won"] = won
    info["player_score"] = player_score
    info["enemy_score"] = enemy_score
    
    return total_reward, steps, info


def main():
    parser = argparse.ArgumentParser(description="Play with trained agent")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between frames (seconds)")
    parser.add_argument("--opponent", type=str, default="simple",
                        choices=["random", "simple", "none"],
                        help="Opponent type")
    parser.add_argument("--win-score", type=int, default=10,
                        help="Score needed to win (use first_to_n mode)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    
    args = parser.parse_args()
    
    # Create environment - use first_to_n for proper game experience
    env_config = EnvConfig(
        enable_opponent=(args.opponent != "none"),
        opponent_type=args.opponent if args.opponent != "none" else "random",
        opponent_delay=5,
        win_mode="first_to_n",
        win_score=args.win_score,
        max_steps=args.max_steps,
    )
    
    env = SlideOrDieEnv(render_mode="human", env_config=env_config)
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(obs_dim, action_dim, DEFAULT_PPO_CONFIG)
    
    # Load model
    if os.path.exists(args.model):
        agent.load(args.model)
    else:
        print(f"Warning: Model not found at {args.model}")
        print("Playing with random policy...")
    
    # Play episodes
    print("\n" + "="*60)
    print("PLAYING SLIDE OR DIE")
    print("="*60)
    
    total_wins = 0
    total_reward = 0
    
    for ep in range(args.episodes):
        reward, steps, info = play_episode(env, agent, args.delay)
        won = info.get("won", False)
        player_score = info.get("player_score", 0)
        enemy_score = info.get("enemy_score", 0)
        truncated = info.get("truncated", False)
        
        total_wins += int(won)
        total_reward += reward
        
        if truncated:
            result = "DRAW (timeout)" if player_score == enemy_score else ("WON!" if won else "LOST")
        else:
            result = "WON!" if won else "LOST"
        
        print(f"Episode {ep + 1}: {result} | "
              f"GREEN(AI): {player_score} vs RED(opponent): {enemy_score} | "
              f"Reward: {reward:.2f} | Steps: {steps}")
    
    print("="*60)
    print(f"Win Rate: {total_wins}/{args.episodes} ({total_wins/args.episodes:.1%})")
    print(f"Average Reward: {total_reward/args.episodes:.2f}")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()

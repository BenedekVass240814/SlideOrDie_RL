"""
Visualization script for V2 trained models.
"""

import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import pygame

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import SlideOrDieEnvV2, EnvConfig
from agent_v2 import PPOAgentV2


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def play_episode(
    env: SlideOrDieEnvV2,
    agent: PPOAgentV2,
    render: bool = True,
    delay: float = 0.1,
    deterministic: bool = True,
) -> dict:
    """Play a single episode."""
    obs, info = env.reset()
    agent.reset_lstm_states(batch_size=1)
    
    done = False
    total_reward = 0
    steps = 0
    
    # Initialize pygame for rendering
    if render:
        pygame.init()
        cell_size = 40
        game = env.game
        width = game.grid_w * cell_size
        height = game.grid_h * cell_size + 60  # Extra space for score
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SlideOrDie V2 - AI Playing")
        font = pygame.font.Font(None, 36)
        clock = pygame.time.Clock()
    
    while not done:
        # Handle pygame events
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return {"quit": True}
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return {"quit": True}
        
        # Get action from agent
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # Render
        if render:
            screen.fill((30, 30, 30))
            
            # Draw game grid
            game = env.game
            for y in range(game.grid_h):
                for x in range(game.grid_w):
                    rect = pygame.Rect(x * cell_size, y * cell_size + 60, cell_size, cell_size)
                    
                    # Map is indexed as [x, y], True = wall
                    is_wall = game.map[x, y]
                    if is_wall:
                        pygame.draw.rect(screen, (100, 100, 100), rect)
                    else:
                        pygame.draw.rect(screen, (50, 50, 50), rect)
                    
                    pygame.draw.rect(screen, (40, 40, 40), rect, 1)
            
            # Draw food - convert from pixel coords to grid, then scale to cell_size
            if game.food_pos:
                fx = game.food_pos.x // game.block_size
                fy = game.food_pos.y // game.block_size
                food_rect = pygame.Rect(fx * cell_size + 5, fy * cell_size + 65, cell_size - 10, cell_size - 10)
                pygame.draw.rect(screen, (255, 255, 0), food_rect)
            
            # Draw player (green - AI)
            px = game.player_pos.x // game.block_size
            py = game.player_pos.y // game.block_size
            player_rect = pygame.Rect(px * cell_size + 2, py * cell_size + 62, cell_size - 4, cell_size - 4)
            pygame.draw.rect(screen, (0, 255, 0), player_rect)
            
            # Draw enemy (red)
            ex = game.enemy_pos.x // game.block_size
            ey = game.enemy_pos.y // game.block_size
            enemy_rect = pygame.Rect(ex * cell_size + 2, ey * cell_size + 62, cell_size - 4, cell_size - 4)
            pygame.draw.rect(screen, (255, 0, 0), enemy_rect)
            
            # Draw scores
            score_text = font.render(
                f"GREEN(AI): {game.score}  vs  RED(Bot): {game.enemy_score}",
                True, (255, 255, 255)
            )
            screen.blit(score_text, (10, 10))
            
            step_text = font.render(f"Step: {steps}/{env.env_config.max_steps}", True, (200, 200, 200))
            screen.blit(step_text, (width - 150, 10))
            
            pygame.display.flip()
            clock.tick(int(1.0 / delay))
    
    # Determine result
    won = info.get("won", False)
    player_score = info.get("player_score", 0)
    enemy_score = info.get("enemy_score", 0)
    
    if render:
        # Show final result
        if won:
            result_text = "AI WINS!"
            color = (0, 255, 0)
        elif player_score < enemy_score:
            result_text = "AI LOSES"
            color = (255, 0, 0)
        else:
            result_text = "DRAW"
            color = (255, 255, 0)
        
        result_surface = font.render(result_text, True, color)
        screen.blit(result_surface, (width // 2 - 50, height // 2))
        pygame.display.flip()
        time.sleep(1.0)
    
    return {
        "reward": total_reward,
        "steps": steps,
        "won": won,
        "player_score": player_score,
        "enemy_score": enemy_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Play SlideOrDie with trained V2 model")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_v2")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("No checkpoint found! Train a model first.")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint to detect architecture
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Detect obs_dim from first layer weight shape
    first_layer_shape = checkpoint["policy_state_dict"]["shared.0.weight"].shape
    detected_obs_dim = first_layer_shape[1]  # Input dimension
    detected_hidden = first_layer_shape[0]   # Hidden dimension
    
    # Detect n_layers by finding max index in shared network
    # Each layer block: Linear(0), LayerNorm(1), ReLU(2), Dropout/Identity(3) = 4 modules
    # So layer 1 uses indices 0-3, layer 2 uses 4-7, layer 3 uses 8-11
    shared_keys = [k for k in checkpoint["policy_state_dict"].keys() if k.startswith("shared.")]
    max_shared_idx = max(int(k.split(".")[1]) for k in shared_keys if k.split(".")[1].isdigit())
    # Correct formula: each layer uses 4 indices, so n_layers = (max_idx // 4) + 1
    detected_n_layers = (max_shared_idx // 4) + 1
    
    # Calculate frame_stack from obs_dim (18 features per frame)
    detected_frame_stack = detected_obs_dim // 18
    
    print(f"Detected from checkpoint: obs_dim={detected_obs_dim}, hidden={detected_hidden}, "
          f"n_layers={detected_n_layers}, frame_stack={detected_frame_stack}")
    
    # Create environment with detected frame_stack
    env_config = EnvConfig(max_steps=300, win_mode="score_compare")
    env = SlideOrDieEnvV2(env_config=env_config, frame_stack=detected_frame_stack)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment obs dim: {obs_dim}, action dim: {action_dim}")
    
    # Create and load agent with detected architecture
    agent = PPOAgentV2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=detected_hidden,
        n_layers=detected_n_layers,
    )
    agent.load(checkpoint_path)
    
    # Play episodes
    wins = 0
    total_reward = 0
    
    print(f"\nPlaying {args.episodes} episodes...")
    print("-" * 40)
    
    for ep in range(args.episodes):
        result = play_episode(
            env=env,
            agent=agent,
            render=not args.no_render,
            delay=args.delay,
            deterministic=not args.stochastic,
        )
        
        if result.get("quit"):
            break
        
        won = result["won"]
        wins += int(won)
        total_reward += result["reward"]
        
        status = "WIN" if won else ("LOSE" if result["player_score"] < result["enemy_score"] else "DRAW")
        print(f"Episode {ep + 1}: {status} | Score: {result['player_score']}-{result['enemy_score']} | Reward: {result['reward']:.2f}")
    
    env.close()
    
    print("-" * 40)
    print(f"Win Rate: {wins}/{args.episodes} ({wins/args.episodes*100:.1f}%)")
    print(f"Avg Reward: {total_reward/args.episodes:.2f}")


if __name__ == "__main__":
    main()

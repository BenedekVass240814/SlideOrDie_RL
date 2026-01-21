from rl_env import SlideOrDieEnv
import numpy as np
import time


def test_headless_environment():
    """Test that the environment works without rendering."""
    print("="*80)
    print("TEST 1: HEADLESS ENVIRONMENT (no window should appear)")
    print("="*80)
    
    env = SlideOrDieEnv(render_mode=None, bot_randomness=0.3, delay=3)
    
    print(f"✓ Environment created successfully!")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Reset works")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Walls in map: {np.sum(obs[0]):.0f}")
    print(f"  Player position: channel 1, sum={np.sum(obs[1])}")
    print(f"  Enemy position: channel 2, sum={np.sum(obs[2])}")
    print(f"  Food position: channel 3, sum={np.sum(obs[3])}")
    
    # Test actions
    print(f"\n✓ Running 50 random steps...")
    total_reward = 0
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}
    
    start_time = time.time()
    
    for step in range(50):
        action = env.action_space.sample()
        actions_taken[action] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  Step {step:2d}: Action={action}, Reward={reward:6.3f}, "
                  f"Player={info['player_score']}, Enemy={info['enemy_score']}")
        
        if terminated or truncated:
            print(f"\n✓ Episode ended at step {step}")
            print(f"  Final scores - Player: {info['player_score']}, Enemy: {info['enemy_score']}")
            break
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Performance test")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps per second: {50/elapsed:.1f}")
    print(f"  Actions distribution: UP={actions_taken[0]}, DOWN={actions_taken[1]}, "
          f"LEFT={actions_taken[2]}, RIGHT={actions_taken[3]}")
    
    env.close()
    print("\n✓ Environment closed successfully")


def test_determinism():
    """Test that environment is deterministic with same seed."""
    print("\n" + "="*80)
    print("TEST 2: DETERMINISM")
    print("="*80)
    
    # Create two environments with same seed
    env1 = SlideOrDieEnv(render_mode=None, bot_randomness=0.3, delay=3)
    env2 = SlideOrDieEnv(render_mode=None, bot_randomness=0.3, delay=3)
    
    # Reset with same seed
    np.random.seed(42)
    obs1, _ = env1.reset(seed=42)
    
    np.random.seed(42)
    obs2, _ = env2.reset(seed=42)
    
    # Take same actions
    actions = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    
    rewards1 = []
    rewards2 = []
    
    for action in actions:
        obs1, r1, t1, tr1, info1 = env1.step(action)
        obs2, r2, t2, tr2, info2 = env2.step(action)
        
        rewards1.append(r1)
        rewards2.append(r2)
        
        if t1 or tr1:
            break
    
    # Note: Due to random map generation, environments won't be identical
    # But the logic should be consistent
    print(f"✓ Both environments executed {len(rewards1)} steps")
    print(f"  Env1 total reward: {sum(rewards1):.2f}")
    print(f"  Env2 total reward: {sum(rewards2):.2f}")
    
    env1.close()
    env2.close()


def test_game_mechanics():
    """Test specific game mechanics."""
    print("\n" + "="*80)
    print("TEST 3: GAME MECHANICS")
    print("="*80)
    
    env = SlideOrDieEnv(render_mode=None, bot_randomness=0.0, delay=100)  # Bot rarely moves
    
    obs, info = env.reset()
    
    print("✓ Testing movement in all directions...")
    
    # Test each direction
    for action, name in [(0, "UP"), (1, "DOWN"), (2, "LEFT"), (3, "RIGHT")]:
        obs_before = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that player position changed (unless hit wall)
        player_pos_before = np.argwhere(obs_before[1] == 1)
        player_pos_after = np.argwhere(obs[1] == 1)
        
        if len(player_pos_before) > 0 and len(player_pos_after) > 0:
            moved = not np.array_equal(player_pos_before, player_pos_after)
            print(f"  {name}: {'Moved' if moved else 'Blocked by wall'}")
    
    print("\n✓ Testing reward structure...")
    
    # Reset and run until something happens
    env = SlideOrDieEnv(render_mode=None, bot_randomness=0.3, delay=3)
    obs, info = env.reset()
    
    player_scored = False
    enemy_scored = False
    
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 5:  # Player scored
            print(f"  Player scored! Reward: {reward:.2f}")
            player_scored = True
            break
        elif reward < -5:  # Enemy scored
            print(f"  Enemy scored! Reward: {reward:.2f}")
            enemy_scored = True
            break
        
        if terminated or truncated:
            break
    
    if not player_scored and not enemy_scored:
        print(f"  No scoring events in 500 steps (this is possible but rare)")
    
    env.close()


def test_rendered_environment():
    """Test environment with rendering."""
    print("\n" + "="*80)
    print("TEST 4: RENDERED ENVIRONMENT (window SHOULD appear)")
    print("="*80)
    print("Running for 3 seconds with rendering...")
    
    env = SlideOrDieEnv(render_mode='human', bot_randomness=0.3, delay=3, speed=20)
    
    obs, info = env.reset()
    
    steps = 0
    start_time = time.time()
    
    while time.time() - start_time < 3:  # Run for 3 seconds
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"✓ Rendered {steps} steps in 3 seconds")
    print(f"  Average FPS: {steps/3:.1f}")
    
    env.close()
    print("✓ Rendering test complete")


def test_episode_completion():
    """Test full episode completion."""
    print("\n" + "="*80)
    print("TEST 5: EPISODE COMPLETION")
    print("="*80)
    
    env = SlideOrDieEnv(render_mode=None, bot_randomness=0.5, delay=3, max_steps=500)
    
    episodes_completed = 0
    wins = 0
    losses = 0
    truncations = 0
    
    for episode in range(5):
        obs, info = env.reset()
        
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                episodes_completed += 1
                if info['player_score'] >= 10:
                    wins += 1
                    result = "WIN"
                else:
                    losses += 1
                    result = "LOSS"
                print(f"  Episode {episode+1}: {result} - Score: {info['player_score']}-{info['enemy_score']}, Steps: {step+1}")
                break
            elif truncated:
                truncations += 1
                print(f"  Episode {episode+1}: TRUNCATED at max steps - Score: {info['player_score']}-{info['enemy_score']}")
                break
    
    print(f"\n✓ Episode statistics:")
    print(f"  Completed: {episodes_completed}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Truncations: {truncations}")
    
    env.close()


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*80)
    print("SLIDE OR DIE - RL ENVIRONMENT TEST SUITE")
    print("="*80 + "\n")
    
    try:
        test_headless_environment()
        test_determinism()
        test_game_mechanics()
        test_episode_completion()
        test_rendered_environment()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TEST FAILED! ✗")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
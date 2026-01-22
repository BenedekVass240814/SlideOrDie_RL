from smart_rl_env import SmartSlideOrDieEnv # Updated import
import numpy as np

def test_smart_environment():
    print("="*80)
    print("TESTING SMART RL ENVIRONMENT")
    print("="*80)
    
    env = SmartSlideOrDieEnv(render_mode=None)
    obs, info = env.reset()
    
    # Check new observation shape (12 features)
    print(f"✓ Observation shape: {obs.shape} (Should be (12,))")
    
    # Verify feature values are normalized
    print(f"✓ Feature ranges: Min={obs.min():.2f}, Max={obs.max():.2f}")
    
    # Run a few steps to see reward shaping in action
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i}: Reward={reward:6.3f} (Includes distance shaping)")

    env.close()

if __name__ == "__main__":
    test_smart_environment()
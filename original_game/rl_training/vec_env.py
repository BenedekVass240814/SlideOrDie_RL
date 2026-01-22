"""
Vectorized environment wrapper for parallel training.
Runs multiple environment instances simultaneously for better GPU utilization.
"""

import numpy as np
from typing import List, Optional, Tuple, Any, Dict
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import gymnasium as gym


class EnvFactory:
    """Picklable environment factory for multiprocessing on Windows."""
    
    def __init__(self, env_class, env_kwargs: Dict[str, Any]):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
    
    def __call__(self):
        return self.env_class(**self.env_kwargs)


def worker(conn: Connection, env_fn):
    """Worker process that runs a single environment."""
    env = env_fn()
    
    while True:
        cmd, data = conn.recv()
        
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                # Auto-reset
                final_obs = obs
                obs, _ = env.reset()
                info["terminal_observation"] = final_obs
            conn.send((obs, reward, done, info))
        
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            conn.send((obs, info))
        
        elif cmd == "close":
            env.close()
            conn.close()
            break
        
        elif cmd == "get_spaces":
            conn.send((env.observation_space, env.action_space))


class VecEnv:
    """
    Vectorized environment that runs multiple instances in parallel.
    
    This allows:
    - Better GPU utilization (batch processing)
    - More samples per second
    - Smoother training curves
    """
    
    def __init__(self, env_fns: List, start_method: str = "spawn"):
        self.n_envs = len(env_fns)
        self.waiting = False
        
        # Create worker processes
        self.parent_conns = []
        self.child_conns = []
        self.processes = []
        
        for env_fn in env_fns:
            parent_conn, child_conn = Pipe()
            process = Process(target=worker, args=(child_conn, env_fn), daemon=True)
            process.start()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            self.processes.append(process)
        
        # Get spaces from first env
        self.parent_conns[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.parent_conns[0].recv()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """Reset all environments."""
        for i, conn in enumerate(self.parent_conns):
            env_seed = seed + i if seed is not None else None
            conn.send(("reset", env_seed))
        
        results = [conn.recv() for conn in self.parent_conns]
        obs = np.stack([r[0] for r in results])
        infos = [r[1] for r in results]
        
        return obs, infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments with given actions."""
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))
        
        results = [conn.recv() for conn in self.parent_conns]
        
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        
        return obs, rewards, dones, infos
    
    def close(self):
        """Close all environments."""
        for conn in self.parent_conns:
            conn.send(("close", None))
        
        for process in self.processes:
            process.join()


class DummyVecEnv:
    """
    Simple vectorized env that runs sequentially (no multiprocessing).
    Useful for debugging or when multiprocessing overhead is too high.
    """
    
    def __init__(self, env_fns: List):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        obs_list = []
        infos = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                info["terminal_observation"] = obs
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos
        )
    
    def close(self):
        for env in self.envs:
            env.close()


def make_vec_env(env_class, n_envs: int = 4, use_multiprocessing: bool = True, 
                 **env_kwargs):
    """
    Create a vectorized environment.
    
    Args:
        env_class: Environment class to instantiate
        n_envs: Number of parallel environments
        use_multiprocessing: Use true multiprocessing (faster) or sequential (debugging)
        **env_kwargs: Arguments to pass to environment constructor
    
    Returns:
        Vectorized environment
    """
    # Use picklable factory class instead of local function
    env_fns = [EnvFactory(env_class, env_kwargs) for _ in range(n_envs)]
    
    if use_multiprocessing:
        return VecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


if __name__ == "__main__":
    # Test vectorized environment
    from env import SlideOrDieEnv
    from config import EnvConfig
    
    print("Testing vectorized environment...")
    
    env_config = EnvConfig(enable_opponent=False, max_steps=100)
    vec_env = make_vec_env(
        SlideOrDieEnv, 
        n_envs=4, 
        use_multiprocessing=True,
        env_config=env_config
    )
    
    print(f"Created {vec_env.n_envs} parallel environments")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Test reset
    obs, infos = vec_env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    
    # Test stepping
    import time
    n_steps = 1000
    start = time.time()
    
    for _ in range(n_steps):
        actions = np.random.randint(0, 4, size=vec_env.n_envs)
        obs, rewards, dones, infos = vec_env.step(actions)
    
    elapsed = time.time() - start
    total_steps = n_steps * vec_env.n_envs
    
    print(f"\nPerformance: {total_steps / elapsed:.0f} steps/second")
    print(f"({vec_env.n_envs} envs × {n_steps} steps in {elapsed:.2f}s)")
    
    vec_env.close()
    print("\nVectorized environment test passed!")

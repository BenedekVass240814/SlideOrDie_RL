"""
Centralized Wandb Configuration.
All training scripts import settings from here for consistency and traceability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    
    # === Project Settings ===
    project: str = "slide-or-die-rl"
    entity: Optional[str] = None  # Your wandb username/team (None = default)
    
    # === Run Settings ===
    run_name: Optional[str] = None  # Auto-generated if None
    run_name_prefix: str = "ppo"  # Prefix for auto-generated names
    tags: List[str] = field(default_factory=lambda: ["ppo", "slide-or-die"])
    notes: str = ""
    
    # === Logging Settings ===
    enabled: bool = True
    log_interval: int = 10  # Log every N rollouts
    save_code: bool = True  # Save code to wandb
    
    # === Metrics to Stream ===
    # Core training metrics (always logged)
    core_metrics: List[str] = field(default_factory=lambda: [
        "train/avg_reward",
        "train/win_rate",
        "train/episodes",
        "train/fps",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/total_loss",
    ])
    
    # Extended metrics (optional, for deeper analysis)
    extended_metrics: List[str] = field(default_factory=lambda: [
        "train/clip_fraction",
        "train/learning_rate",
        "train/entropy_coef",
        "train/explained_variance",
        "train/approx_kl",
        "episode/avg_length",
        "episode/max_reward",
        "episode/min_reward",
        "game/player_score_avg",
        "game/enemy_score_avg",
        "game/draws",
        "game/map_changes",
        "perf/gpu_memory_mb",
        "perf/steps_per_second",
    ])
    
    # Custom metrics (user-defined)
    custom_metrics: List[str] = field(default_factory=list)
    
    def generate_run_name(self, **kwargs) -> str:
        """Generate a descriptive run name from training config."""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        parts = [self.run_name_prefix]
        
        # Add config values to name
        if "version" in kwargs:
            parts.append(f"v{kwargs['version']}")
        if "n_envs" in kwargs:
            parts.append(f"{kwargs['n_envs']}env")
        if "frame_stack" in kwargs:
            parts.append(f"fs{kwargs['frame_stack']}")
        if "hidden_size" in kwargs:
            parts.append(f"{kwargs['hidden_size']}h")
        if "map_change_episodes" in kwargs:
            parts.append(f"mc{kwargs['map_change_episodes']}")
        
        parts.append(timestamp)
        
        return "_".join(parts)
    
    def get_all_metrics(self) -> List[str]:
        """Get all metrics to be logged."""
        return self.core_metrics + self.extended_metrics + self.custom_metrics


@dataclass  
class MapConfig:
    """Map generation and curriculum settings."""
    
    # === Fixed Map Training ===
    use_fixed_map: bool = True  # Use single map instead of random each reset
    map_change_episodes: int = 500  # Generate new map every N episodes
    map_change_on_plateau: bool = True  # Change map if win rate plateaus
    plateau_threshold: float = 0.05  # Win rate change threshold for plateau
    plateau_window: int = 100  # Episodes to check for plateau
    
    # === Map Pool (for variety without full randomness) ===
    use_map_pool: bool = False  # Use a fixed pool of maps
    map_pool_size: int = 10  # Number of maps in the pool
    
    # === Map Difficulty ===
    initial_wall_density: float = 0.15  # Starting wall density
    max_wall_density: float = 0.25  # Maximum wall density
    wall_density_increase_rate: float = 0.01  # Increase per map change
    
    # === Seed Control ===
    map_seed: Optional[int] = None  # Fixed seed for reproducibility (None = random)


# === Default Instances ===
DEFAULT_WANDB_CONFIG = WandbConfig()
DEFAULT_MAP_CONFIG = MapConfig()


def init_wandb(
    wandb_config: WandbConfig,
    training_config: dict,
    resume: bool = False,
) -> Optional[object]:
    """
    Initialize wandb with the given configuration.
    
    Args:
        wandb_config: WandbConfig instance
        training_config: Dict of training hyperparameters to log
        resume: Whether to resume a previous run
    
    Returns:
        wandb run object or None if disabled/unavailable
    """
    if not wandb_config.enabled:
        print("Wandb logging disabled")
        return None
    
    try:
        import wandb
    except ImportError:
        print("wandb not installed - training will continue without remote logging")
        return None
    
    # Generate run name if not provided
    run_name = wandb_config.run_name
    if run_name is None:
        run_name = wandb_config.generate_run_name(**training_config)
    
    # Initialize wandb
    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=run_name,
        tags=wandb_config.tags,
        notes=wandb_config.notes,
        config=training_config,
        save_code=wandb_config.save_code,
        resume="allow" if resume else None,
    )
    
    print(f"Wandb initialized: {wandb_config.project}/{run_name}")
    print(f"  Dashboard: {run.url}")
    
    return run


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    wandb_run = None,
):
    """
    Log metrics to wandb.
    
    Args:
        metrics: Dict of metric name -> value
        step: Current training step
        wandb_run: Wandb run object (if None, tries to use global wandb)
    """
    if wandb_run is None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass
    else:
        wandb_run.log(metrics, step=step)

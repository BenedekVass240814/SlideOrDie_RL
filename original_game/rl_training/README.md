# Slide or Die - Efficient RL Training

A more efficient reinforcement learning setup for the Slide or Die game.

## Key Improvements Over Original RL Setup

### 1. **Compact Feature-Based Observations** (vs. Full Grid CNN)
- Original: 4-channel grid image (4 × 18 × 32 = 2,304 values)
- New: 17-feature vector with relative positions and local obstacles
- **Result**: Faster training, smaller network, better generalization

### 2. **PPO Algorithm** (vs. DQN)
- More stable training with clipped surrogate objective
- Better sample efficiency with GAE advantage estimation
- Natural support for continuous training

### 3. **Smart Reward Shaping**
- Distance-based intermediate rewards (closer to food = small reward)
- Wall collision penalties
- Win/lose bonuses

### 4. **Curriculum Learning**
- Stage 0: No opponent (learn basic food collection)
- Stage 1: Random opponent (learn competition)
- Stage 2: Simple greedy opponent (harder competition)
- Auto-advances when win rate reaches threshold

### 5. **No Pathfinding Opponent**
- Simple opponents that the model can learn to beat
- The agent develops its own navigation strategy

## Files

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters and configuration dataclasses |
| `game_core.py` | Minimal game logic (no rendering, optimized for speed) |
| `env.py` | Gymnasium environment wrapper |
| `agent.py` | PPO agent and neural network |
| `train.py` | Main training script with curriculum learning |
| `play.py` | Visualize trained agent |

## Usage

### Training
```bash
# Basic training (500k steps with curriculum)
python train.py

# Custom settings
python train.py --timesteps 1000000 --lr 0.0003

# Without curriculum learning
python train.py --no-curriculum
```

### Playing with Trained Agent
```bash
# Play 5 episodes with best model
python play.py

# Custom settings
python play.py --model checkpoints/best_model.pth --episodes 10 --opponent simple
```

## Observation Space (17 features)

| Feature | Description |
|---------|-------------|
| 0-1 | Player position (normalized x, y) |
| 2-3 | Food relative position (dx, dy from player) |
| 4-5 | Enemy relative position (dx, dy from player) |
| 6 | Distance to food (normalized) |
| 7 | Distance to enemy (normalized) |
| 8-11 | Local obstacles (UP, DOWN, LEFT, RIGHT) |
| 12-15 | Current direction (one-hot) |
| 16 | Score difference (normalized) |

## Action Space

| Action | Direction |
|--------|-----------|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

## Expected Training Time

- **Stage 0** (no opponent): ~50k steps
- **Stage 1** (random opponent): ~100k steps  
- **Stage 2** (simple opponent): ~200k steps
- **Total**: ~350-500k steps for competent agent

With a modern GPU, this should take 15-30 minutes.

## Tips for Better Results

1. **Start with curriculum**: It helps the agent learn fundamentals
2. **Monitor win rate**: Should increase over time
3. **Adjust entropy coefficient**: Lower (0.005) for exploitation, higher (0.02) for exploration
4. **Increase n_steps**: More steps per update = more stable but slower

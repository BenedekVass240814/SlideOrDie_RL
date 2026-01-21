import numpy as np
from enum import Enum
from collections import namedtuple
from modules.map_generator import MapGenerator

BLOCK_SIZE = 50


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SlideOrDieCore:
    """
    Core game logic without any rendering or sound.
    Perfect for RL training and can be wrapped with rendering later.
    """
    
    def __init__(self, bot_randomness: float = 0.5, chase: bool = False, 
                 delay: int = 5, w=1600, h=900, map_generator=None):
        self.chase = chase
        self.delay = delay
        self.bot_randomness = bot_randomness
        self.h = h
        self.w = w
        
        # Generate or use provided map
        if map_generator is None:
            generator = MapGenerator(w=w, h=h)
        else:
            generator = map_generator
            
        self.map = generator.map
        self.spawn_point = generator.spawn_point
        self.enemy_spawn = generator.enemy_spawn
        
        # Initialize game state
        self.reset()
        
        # Bot is optional - will be set by wrapper if needed
        self.bot = None
        self.bot_enabled = False
    
    def enable_bot(self, Bot):
        """Enable bot opponent (optional)."""
        from bots.bot03 import Bot as BotClass
        self.bot = BotClass(self.map, randomness=self.bot_randomness)
        self.bot_enabled = True
    
    def reset(self):
        """Reset game to initial state."""
        self.timer = 0
        self.direction = Direction.DOWN
        self.head = Point(self.spawn_point.x * BLOCK_SIZE, self.spawn_point.y * BLOCK_SIZE)
        self.enemy = Point(self.enemy_spawn.x * BLOCK_SIZE, self.enemy_spawn.y * BLOCK_SIZE)
        self.score = 0
        self.enemy_score = 0
        self.food = None
        self.place_food()
        self.enemy_direction = 0
        
        return self.get_state()
    
    def get_state(self):
        """Return current game state as a dictionary."""
        return {
            'head': self.head,
            'enemy': self.enemy,
            'food': self.food,
            'score': self.score,
            'enemy_score': self.enemy_score,
            'direction': self.direction,
            'map': self.map
        }
    
    def place_food(self):
        """Place food at random valid location."""
        x = np.random.randint(0, self.map.shape[0])
        y = np.random.randint(0, self.map.shape[1])
        while self.map[x, y]:
            x = np.random.randint(0, self.map.shape[0])
            y = np.random.randint(0, self.map.shape[1])
        
        self.food = Point(x * BLOCK_SIZE, y * BLOCK_SIZE)
    
    def respawn(self):
        """Respawn player at random valid location."""
        x = np.random.randint(0, self.map.shape[0])
        y = np.random.randint(0, self.map.shape[1])
        while self.map[x, y] or Point(x, y) == self.enemy:
            x = np.random.randint(0, self.map.shape[0])
            y = np.random.randint(0, self.map.shape[1])
        
        self.head = Point(x * BLOCK_SIZE, y * BLOCK_SIZE)
    
    def set_direction(self, direction: Direction):
        """Set player direction."""
        self.direction = direction
    
    def _is_collision(self, pt):
        """Check if point collides with wall or boundary."""
        rect_head_x = pt.x
        rect_head_y = pt.y
        
        # Check boundary collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Check wall collision
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j]:
                    wall_x = i * BLOCK_SIZE
                    wall_y = j * BLOCK_SIZE
                    
                    # Simple AABB collision
                    if (rect_head_x < wall_x + BLOCK_SIZE and
                        rect_head_x + BLOCK_SIZE > wall_x and
                        rect_head_y < wall_y + BLOCK_SIZE and
                        rect_head_y + BLOCK_SIZE > wall_y):
                        return True
        
        return False
    
    def _move(self, check=False, target=None, direction=None):
        """Calculate next position based on direction."""
        if target is None:
            x = self.head.x
            y = self.head.y
        else:
            x = target.x
            y = target.y
        
        if direction is None:
            direction = self.direction
        else:
            dirs = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
            direction = dirs[direction - 1]
        
        if direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        else:
            x += BLOCK_SIZE
        
        if not check:
            if target is None:
                self.head = Point(x, y)
            else:
                self.enemy = Point(x, y)
        
        return Point(x, y)
    
    def step(self, player_action: Direction = None):
        """
        Execute one game step.
        
        Args:
            player_action: Direction for player to move (if None, continues current direction)
        
        Returns:
            dict with keys: 'player_collected', 'enemy_collected', 'player_caught', 'done'
        """
        events = {
            'player_collected': False,
            'enemy_collected': False,
            'player_caught': False,
            'done': False
        }
        
        # Update player direction if action provided
        if player_action is not None:
            self.direction = player_action
        
        # Increment timer for enemy
        self.timer += 1
        
        # Move player
        if not self._is_collision(self._move(check=True)):
            self._move()
        
        # Check if player collected food
        if self.food == self.head:
            self.place_food()
            self.score += 1
            events['player_collected'] = True
        
        # Enemy AI (if enabled and timer allows)
        if self.bot_enabled and self.timer >= self.delay:
            self.timer = 0
            target = self.head if self.chase else self.food
            bot_action = self.bot.forward(self.enemy, target)
            bot_step = self._move(check=True, target=self.enemy, direction=bot_action)
            
            if not self._is_collision(bot_step):
                self._move(target=self.enemy, direction=bot_action)
                self.enemy_direction = bot_action
        
        # Check if enemy collected food (non-chase mode)
        if self.enemy == self.food and not self.chase:
            self.place_food()
            self.enemy_score += 1
            events['enemy_collected'] = True
        
        # Check if enemy caught player (chase mode)
        elif self.enemy == self.head and self.chase:
            self.enemy_score += 1
            self.respawn()
            events['player_caught'] = True
        
        # Check win/loss conditions
        if self.score >= 10 or self.enemy_score >= 10:
            events['done'] = True
        
        return events
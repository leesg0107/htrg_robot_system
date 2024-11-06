import numpy as np
from gym import spaces

class BaseEnv:
    def __init__(self, config):
        self.grid_size = config['grid_size']
        self.max_steps = config['max_steps']
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.grid_size, self.grid_size, config['n_channels']),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32  # [direction, speed]
        )

import numpy as np
from gym import spaces
from typing import Dict, Tuple, List

class BaseEnv:
    """
    Base environment class that defines the common interface and functionality
    for all environment implementations.
    """
    def __init__(self, config: Dict):
        """
        Initialize the base environment.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - grid_size: Size of the environment grid
                - max_steps: Maximum steps per episode
                - n_channels: Number of observation channels
                - n_catchers: Number of catcher agents
                - n_runners: Number of runner agents
        """
        self.grid_size = config['grid_size']
        self.max_steps = config['max_steps']
        self.n_catchers = config['n_catchers']
        self.n_runners = config['n_runners']
        self.current_step = 0
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, config['n_channels']),
            dtype=np.float32
        )
        
        # Continuous action space for [direction, speed]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self) -> Dict:
        """Reset the environment to initial state."""
        self.current_step = 0
        raise NotImplementedError
        
    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            actions (Dict): Actions for each agent
            
        Returns:
            Tuple containing:
            - observations (Dict): Each agent's observation
            - rewards (Dict): Each agent's reward
            - dones (Dict): Done flags for each agent
            - info (Dict): Additional information
        """
        self.current_step += 1
        raise NotImplementedError
    
    def get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for specific agent."""
        raise NotImplementedError
    
    def _check_done(self) -> bool:
        """Check if episode should end."""
        if self.current_step >= self.max_steps:
            return True
        return False
    
    def render(self):
        """Render the environment."""
        raise NotImplementedError

    def close(self):
        """Clean up environment resources"""
        pass

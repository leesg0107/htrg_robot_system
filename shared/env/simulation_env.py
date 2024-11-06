import numpy as np
from typing import Dict, Tuple, List
from env.base_env import BaseEnv

class SimulationEnv(BaseEnv):
    """
    Simulation environment for the catch-runner scenario.
    Implements the actual game logic and physics.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Additional simulation-specific parameters
        self.collision_radius = config.get('collision_radius', 2.0)
        self.capture_radius = config.get('capture_radius', 3.0)
        self.agent_speed = config.get('agent_speed', 1.0)
        
        # State variables
        self.catcher_positions = {}  # Dict to store catcher positions
        self.runner_positions = {}   # Dict to store runner positions
        self.obstacles = []          # List to store obstacle positions
        
    def reset(self) -> Dict:
        """Reset the environment and return initial observations."""
        self.current_step = 0
        
        # Reset agent positions randomly
        self._reset_positions()
        
        # Generate initial observations for all agents
        observations = {}
        for catcher_id in range(self.n_catchers):
            observations[f'catcher_{catcher_id}'] = self.get_observation(f'catcher_{catcher_id}')
        for runner_id in range(self.n_runners):
            observations[f'runner_{runner_id}'] = self.get_observation(f'runner_{runner_id}')
            
        return observations
    
    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one simulation step."""
        self.current_step += 1
        
        # Update positions based on actions
        self._update_positions(actions)
        
        # Get new observations
        observations = {}
        for catcher_id in range(self.n_catchers):
            observations[f'catcher_{catcher_id}'] = self.get_observation(f'catcher_{catcher_id}')
        for runner_id in range(self.n_runners):
            observations[f'runner_{runner_id}'] = self.get_observation(f'runner_{runner_id}')
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check if episode is done
        dones = self._check_termination()
        
        # Additional info
        info = {
            'captures': self._count_captures(),
            'collisions': self._count_collisions()
        }
        
        return observations, rewards, dones, info
    
    def get_observation(self, agent_id: str) -> np.ndarray:
        """Generate observation for specific agent."""
        # Create grid-like sensor observation
        obs = np.zeros((self.grid_size, self.grid_size, 3))  # 3 channels: catchers, runners, obstacles
        
        # Fill observation channels
        if 'catcher' in agent_id:
            agent_pos = self.catcher_positions[agent_id]
        else:
            agent_pos = self.runner_positions[agent_id]
            
        # Add agents and obstacles to observation (with partial observability)
        self._add_visible_agents_to_obs(obs, agent_pos)
        self._add_obstacles_to_obs(obs)
        
        return obs
    
    def _reset_positions(self):
        """Reset agent positions randomly while avoiding collisions."""
        # Initialize empty positions
        self.catcher_positions = {}
        self.runner_positions = {}
        
        # Place catchers
        for i in range(self.n_catchers):
            while True:
                pos = np.random.uniform(0, self.grid_size, 2)
                if self._is_valid_position(pos):
                    self.catcher_positions[f'catcher_{i}'] = pos
                    break
        
        # Place runners
        for i in range(self.n_runners):
            while True:
                pos = np.random.uniform(0, self.grid_size, 2)
                if self._is_valid_position(pos):
                    self.runner_positions[f'runner_{i}'] = pos
                    break
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """
        Check if position is valid (no collisions with other agents or obstacles).
        """
        # Check boundaries
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
            
        # Check collision with obstacles
        for obs_pos in self.obstacles:
            if np.linalg.norm(pos - obs_pos) < self.collision_radius:
                return False
                
        # Check collision with other agents
        for catcher_pos in self.catcher_positions.values():
            if np.linalg.norm(pos - catcher_pos) < self.collision_radius:
                return False
                
        for runner_pos in self.runner_positions.values():
            if np.linalg.norm(pos - runner_pos) < self.collision_radius:
                return False
                
        return True
    
    def _update_positions(self, actions: Dict):
        """
        Update agent positions based on actions.
        Actions are in the format: [direction, speed]
        """
        # Update catcher positions
        for catcher_id in range(self.n_catchers):
            action = actions[f'catcher_{catcher_id}']
            current_pos = self.catcher_positions[f'catcher_{catcher_id}']
            
            # Convert action to movement
            direction = action[0] * 2 * np.pi  # Convert to radians
            speed = (action[1] + 1) / 2 * self.agent_speed  # Normalize to [0, agent_speed]
            
            # Calculate new position
            delta_pos = np.array([
                speed * np.cos(direction),
                speed * np.sin(direction)
            ])
            new_pos = current_pos + delta_pos
            
            # Check if new position is valid
            if self._is_valid_position(new_pos):
                self.catcher_positions[f'catcher_{catcher_id}'] = new_pos
                
        # Update runner positions
        for runner_id in range(self.n_runners):
            action = actions[f'runner_{runner_id}']
            current_pos = self.runner_positions[f'runner_{runner_id}']
            
            # Convert action to movement
            direction = action[0] * 2 * np.pi
            speed = (action[1] + 1) / 2 * self.agent_speed
            
            # Calculate new position
            delta_pos = np.array([
                speed * np.cos(direction),
                speed * np.sin(direction)
            ])
            new_pos = current_pos + delta_pos
            
            # Check if new position is valid
            if self._is_valid_position(new_pos):
                self.runner_positions[f'runner_{runner_id}'] = new_pos
    
    def _calculate_rewards(self) -> Dict:
        """
        Calculate rewards for all agents.
        Returns dictionary with rewards for each agent.
        """
        rewards = {}
        
        # Calculate base rewards
        for catcher_id in range(self.n_catchers):
            rewards[f'catcher_{catcher_id}'] = -0.01  # Small negative reward per step
            
        for runner_id in range(self.n_runners):
            rewards[f'runner_{runner_id}'] = 0.01  # Small positive reward for surviving
            
        # Check captures
        for catcher_id, catcher_pos in self.catcher_positions.items():
            for runner_id, runner_pos in self.runner_positions.items():
                distance = np.linalg.norm(catcher_pos - runner_pos)
                
                if distance < self.capture_radius:
                    # Reward for successful capture
                    rewards[catcher_id] += 10.0
                    rewards[runner_id] -= 10.0
                    
        # Reward for collision avoidance
        collision_penalty = -5.0
        for catcher_id, pos1 in self.catcher_positions.items():
            for catcher_id2, pos2 in self.catcher_positions.items():
                if catcher_id != catcher_id2:
                    if np.linalg.norm(pos1 - pos2) < self.collision_radius:
                        rewards[catcher_id] += collision_penalty
                        rewards[catcher_id2] += collision_penalty
                        
        return rewards
    
    def _check_termination(self) -> Dict:
        """Check termination conditions for all agents."""
        dones = {
            'all_done': self._check_done()
        }
        for catcher_id in range(self.n_catchers):
            dones[f'catcher_{catcher_id}'] = dones['all_done']
        for runner_id in range(self.n_runners):
            dones[f'runner_{runner_id}'] = dones['all_done']
        return dones
    
    def _count_captures(self) -> int:
        """Count number of captures in current state."""
        captures = 0
        for catcher_pos in self.catcher_positions.values():
            for runner_pos in self.runner_positions.values():
                if np.linalg.norm(catcher_pos - runner_pos) < self.capture_radius:
                    captures += 1
        return captures
    
    def _count_collisions(self) -> int:
        """Count number of collisions in current state."""
        collisions = 0
        # Check catcher-catcher collisions
        catcher_positions = list(self.catcher_positions.values())
        for i in range(len(catcher_positions)):
            for j in range(i + 1, len(catcher_positions)):
                if np.linalg.norm(catcher_positions[i] - catcher_positions[j]) < self.collision_radius:
                    collisions += 1
                    
        # Check catcher-obstacle collisions
        for catcher_pos in catcher_positions:
            for obstacle_pos in self.obstacles:
                if np.linalg.norm(catcher_pos - obstacle_pos) < self.collision_radius:
                    collisions += 1
                    
        return collisions
    
    def _add_visible_agents_to_obs(self, obs: np.ndarray, agent_pos: np.ndarray):
        """Add visible agents to observation grid."""
        view_range = self.grid_size // 2
        
        # Add catchers to channel 0
        for pos in self.catcher_positions.values():
            if np.linalg.norm(pos - agent_pos) < view_range:
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    obs[grid_x, grid_y, 0] = 1
                    
        # Add runners to channel 1
        for pos in self.runner_positions.values():
            if np.linalg.norm(pos - agent_pos) < view_range:
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    obs[grid_x, grid_y, 1] = 1
    
    def _add_obstacles_to_obs(self, obs: np.ndarray):
        """Add obstacles to observation grid."""
        # Add obstacles to channel 2
        for pos in self.obstacles:
            grid_x = int(pos[0])
            grid_y = int(pos[1])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                obs[grid_x, grid_y, 2] = 1

    def close(self):
        """Clean up environment resources"""
        super().close()
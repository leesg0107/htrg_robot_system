import numpy as np
import torch
from typing import Dict, Tuple
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state: torch.Tensor, action: torch.Tensor, 
              reward: torch.Tensor, next_state: torch.Tensor, done: bool):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """배치 크기만큼 경험 샘플링"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (torch.stack(states),
                torch.stack(actions),
                torch.stack(rewards),
                torch.stack(next_states),
                torch.tensor(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

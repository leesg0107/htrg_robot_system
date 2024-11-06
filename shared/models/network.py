import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 행동 범위 [-1, 1]
        )
        
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return self.network(state)

class AttentionNetwork(nn.Module):
    def __init__(self, state_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(state_dim, num_heads)
        
    def forward(self, states):
        # states shape: (batch_size, num_agents, state_dim)
        attn_output, attn_weights = self.attention(states, states, states)
        return attn_output, attn_weights

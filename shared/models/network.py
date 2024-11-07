import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # 3D 입력을 1D로 평탄화
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        정책 함수 계산
        Args:
            x: 입력 상태 텐서
        Returns:
            torch.Tensor: 행동 평균
        """
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # 3D 입력을 1D로 평탄화
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        가치 함수 계산
        Args:
            x: 입력 상태 텐서
        Returns:
            torch.Tensor: 상태 가치
        """
        return self.network(x)

class AttentionNetwork(nn.Module):
    def __init__(self, state_dim, num_heads=4):
        super().__init__()
        self.embed_dim = 256  # num_heads(4)로 나누어 떨어지는 값
        
        self.input_projection = nn.Linear(state_dim, self.embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, states):
        batch_size, num_agents, _ = states.shape
        
        projected_states = self.input_projection(states)  # (batch_size, num_agents, embed_dim)
        
        attn_output, attn_weights = self.attention(
            projected_states,
            projected_states,
            projected_states
        )
        
        return attn_output, attn_weights

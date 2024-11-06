import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from network import PolicyNetwork, ValueNetwork, AttentionNetwork

class MAPOCA:
    def __init__(self, config: Dict):
        """
        MA-POCA (Multi-Agent POsthumous Credit Assignment) 초기화
        Args:
            config: 설정 파라미터
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 네트워크 초기화
        self.policy_net = PolicyNetwork(
            config['state_dim'], 
            config['action_dim']
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            config['state_dim']
        ).to(self.device)
        
        self.attention_net = AttentionNetwork(
            config['state_dim']
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get('policy_lr', 3e-4))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.get('value_lr', 1e-3))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lambda_ = config.get('lambda', 0.95)
        self.epsilon = config.get('epsilon', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        상태에 따른 행동 선택
        """
        with torch.no_grad():
            action = self.policy_net(state)
        return action
        
    def compute_gae(self, rewards: List[float], values: torch.Tensor, 
                   dones: List[bool]) -> torch.Tensor:
        """
        Generalized Advantage Estimation 계산
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, device=self.device)
        
    def credit_assignment(self, states: torch.Tensor, actions: torch.Tensor, 
                         rewards: torch.Tensor) -> torch.Tensor:
        """
        Credit Assignment using Attention Mechanism
        """
        # Attention을 통한 에이전트 간 관계 계산
        attn_output, attn_weights = self.attention_net(states)
        
        # 각 에이전트의 기여도에 따른 보상 계산
        credited_rewards = rewards.unsqueeze(-1) * attn_weights
        
        return credited_rewards
        
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Policy Update using PPO
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_probs = batch['old_probs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 1. GAE 계산
        values = self.value_net(states)
        advantages = self.compute_gae(rewards, values, dones)
        
        # 2. Credit Assignment
        credited_rewards = self.credit_assignment(states, actions, rewards)
        
        # 3. Policy Update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.n_epochs):
            # 현재 정책의 행동 확률
            new_probs = self.policy_net(states)
            
            # PPO ratio
            ratio = torch.exp(new_probs - old_probs)
            
            # PPO loss 계산
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss 계산
            value_pred = self.value_net(states)
            value_loss = F.mse_loss(value_pred, credited_rewards)
            
            # Optimization step
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            policy_loss.backward()
            value_loss.backward()
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
        return {
            'policy_loss': total_policy_loss / self.n_epochs,
            'value_loss': total_value_loss / self.n_epochs
        }

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
from base_agent import BaseAgent

class Catcher(BaseAgent):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Catcher 특화 파라미터
        self.capture_radius = config.get('capture_radius', 1.5)
        self.collision_penalty = config.get('collision_penalty', -5.0)
        
        # 정책 신경망 초기화
        self.policy_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.grid_size * self.grid_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
    def act(self, observation: np.ndarray) -> np.ndarray:
        """포획을 위한 행동 선택"""
        obs = self._preprocess_observation(observation)
        
        with torch.no_grad():
            action_logits = self.policy_net(obs)
            # 행동을 [-1, 1] 범위로 제한
            action = torch.tanh(action_logits)
            
        return action.cpu().numpy()[0]
        
    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """경험으로부터 학습"""
        # 배치 데이터 추출
        states = torch.FloatTensor(experience['states']).to(self.device)
        actions = torch.FloatTensor(experience['actions']).to(self.device)
        rewards = torch.FloatTensor(experience['rewards']).to(self.device)
        next_states = torch.FloatTensor(experience['next_states']).to(self.device)
        dones = torch.FloatTensor(experience['dones']).to(self.device)
        
        # 정책 그래디언트 계산
        self.optimizer.zero_grad()
        
        action_logits = self.policy_net(states)
        action_probs = torch.tanh(action_logits)
        
        # MSE 손실 사용 (간단한 버전)
        loss = nn.MSELoss()(action_probs, actions)
        
        # 역전파 및 최적화
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item()
        }
        
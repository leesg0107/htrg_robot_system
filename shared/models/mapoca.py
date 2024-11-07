import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple
from shared.models.network import PolicyNetwork, ValueNetwork, AttentionNetwork
from shared.utils.replay_buffer import ReplayBuffer
import os
import numpy as np

class MAPOCA:
    def __init__(self, config):
        """
        MAPOCA 초기화
        Args:
            config: 설정 딕셔너리
        """
        # 설정 저장
        self.config = config
        
        # 기본 설정
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_agents = config['n_agents']
        self.device = config['device']
        self.action_std = config['action_std']
        
        # 학습 설정
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.tau = config['tau']
        
        # 네트워크 초기화
        self.policy_net = PolicyNetwork(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 옵티마이저 초기화
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=self.learning_rate
        )
        
        # 리플레이 버퍼 초기화
        self.buffer = {
            agent_id: {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            }
            for agent_id in range(self.n_agents)
        }
        
        # 학습 모드 설정
        self.training = True
    
    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """
        경험 저장
        Args:
            agent_id: 에이전트 ID
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 종료 여부
        """
        # 버퍼가 가득 찬 경우 오래된 데이터 제거
        if len(self.buffer[agent_id]['states']) >= self.buffer_size:
            for key in self.buffer[agent_id].keys():
                self.buffer[agent_id][key].pop(0)
        
        # 새로운 경험 저장
        self.buffer[agent_id]['states'].append(state)
        self.buffer[agent_id]['actions'].append(action)
        self.buffer[agent_id]['rewards'].append(reward)
        self.buffer[agent_id]['next_states'].append(next_state)
        self.buffer[agent_id]['dones'].append(done)
    
    def is_ready_to_update(self) -> bool:
        """학습 가능 여부 확인"""
        # 모든 에이전트의 버퍼가 배치 크기 이상 채워졌는지 확인
        return all(
            len(self.buffer[agent_id]['states']) >= self.batch_size
            for agent_id in self.buffer.keys()
        )
    
    def sample_batch(self, agent_id):
        """
        배치 샘플링
        Args:
            agent_id: 에이전트 ID
        Returns:
            dict: 샘플링된 배치 데이터
        """
        buffer = self.buffer[agent_id]
        buffer_size = len(buffer['states'])
        
        # 랜덤 인덱스 선택
        indices = np.random.choice(buffer_size, self.batch_size, replace=False)
        
        # numpy 배열로 변환 후 텐서로 변환
        return {
            'states': torch.FloatTensor(np.array([buffer['states'][i] for i in indices])).to(self.device),
            'actions': torch.FloatTensor(np.array([buffer['actions'][i] for i in indices])).to(self.device),
            'rewards': torch.FloatTensor(np.array([buffer['rewards'][i] for i in indices])).to(self.device),
            'next_states': torch.FloatTensor(np.array([buffer['next_states'][i] for i in indices])).to(self.device),
            'dones': torch.FloatTensor(np.array([buffer['dones'][i] for i in indices])).to(self.device)
        }
    
    def select_action(self, state: torch.Tensor, agent_id: str) -> torch.Tensor:
        """
        상태에 대한 행동 선택
        Args:
            state: 현재 상태 (any shape)
            agent_id: 에이전트 ID
        Returns:
            torch.Tensor: 선택된 행동 (action_dim,)
        """
        with torch.no_grad():
            # 상태가 3D array인 경우 처리
            if state.dim() > 2:
                state = state.unsqueeze(0) if state.dim() == 3 else state
            
            # 정책 네트워크로 행동 평균 계산
            action_mean = self.policy_net(state)
            
            if self.training:
                # 학습 모드: 탐색을 위한 노이즈 추가
                action_std = torch.ones_like(action_mean) * self.action_std
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.sample()
            else:
                # 평가 모드: 평균값 사용
                action = action_mean
            
            # 행동 클리핑
            action = torch.clamp(action, -1.0, 1.0)
            
            # 배치 차원이 있다면 제거
            if action.dim() > 1:
                action = action.squeeze(0)
            
            return action
    
    def train(self, mode=True):
        """
        학습/평가 모드 설정
        Args:
            mode (bool): True면 학습 모드, False면 평가 모드
        """
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> torch.Tensor:
        """
        Generalized Advantage Estimation 계산
        """
        advantages = []
        gae = 0
        
        # dones를 float 타입으로 변환
        dones = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.stack(advantages)
        
    def credit_assignment(self, states: torch.Tensor, actions: torch.Tensor, 
                         rewards: torch.Tensor) -> torch.Tensor:
        """
        Credit Assignment using Attention Mechanism
        """
        with torch.no_grad():  # credit assignment는 gradient 계산이 필요 없음
            attn_output, attn_weights = self.attention_net(states)
            rewards_expanded = rewards.unsqueeze(-1)
            credited_rewards = rewards_expanded * attn_weights.mean(dim=-1, keepdim=True)
            return credited_rewards.squeeze(-1)
        
    def update(self):
        """모델 업데이트"""
        if not self.is_ready_to_update():
            return None
        
        # 배치 데이터 샘플링
        batch = self.sample_batch(0)  # 첫 번째 에이전트의 배치
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 현재 상태의 가치 추정
        current_values = self.value_net(states).squeeze()
        
        # 다음 상태의 가치 추정
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        # 가치 함수 업데이트
        value_loss = F.mse_loss(current_values, target_values)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 정책 함수 업데이트
        action_mean = self.policy_net(states)
        action_std = torch.ones_like(action_mean) * self.action_std
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        advantages = (target_values - current_values).detach()
        
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_value': current_values.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def save_checkpoint(self, path: str):
        """모델 체크포인트 저장"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'config': self.config
        }
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 체크���인트 저장
        torch.save(checkpoint, path + '.pt')
        print(f"Checkpoint saved: {path}.pt")
    
    def load_checkpoint(self, path: str):
        """모델 체크포인트 로드"""
        if not os.path.exists(path + '.pt'):
            print(f"No checkpoint found at {path}.pt")
            return False
            
        checkpoint = torch.load(path + '.pt')
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        print(f"Checkpoint loaded: {path}.pt")
        return True

from collections import deque
import numpy as np
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn

class BaseAgent:
    def __init__(self, config: Dict):
        """
        기본 에이전트 초기화
        Args:
            config: 에이전트 설정
        """
        self.grid_size = config.get('grid_size', 10)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = 2  # [direction, speed]
        
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        관찰을 바탕으로 행동 선택
        Args:
            observation: (grid_size, grid_size, channels) 형태의 관찰
        Returns:
            action: [direction, speed] 형태의 행동
        """
        raise NotImplementedError
        
    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        경험으로부터 학습
        Args:
            experience: 학습에 필요한 데이터
        Returns:
            metrics: 학습 관련 지표들
        """
        raise NotImplementedError
        
    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """관찰 전처리"""
        # numpy array를 torch tensor로 변환
        obs = torch.FloatTensor(observation).to(self.device)
        
        # 차원 순서 변경: (H, W, C) -> (C, H, W)
        obs = obs.permute(2, 0, 1)
        
        # batch dimension 추가
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # (1, C, H, W)
        
        return obs

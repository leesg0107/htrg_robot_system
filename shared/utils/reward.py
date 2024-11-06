import numpy as np
from typing import Dict, List, Tuple

class RewardCalculator:
    def __init__(self, config: Dict):
        """
        보상 계산기 초기화
        Args:
            config: 설정 파라미터
                - collision_threshold: 충돌 판정 거리
                - capture_threshold: 포획 판정 거리
        """
        self.collision_threshold = config.get('collision_threshold', 1.0)
        self.capture_threshold = config.get('capture_threshold', 2.0)
        
    def calculate_rewards(self, state: Dict, next_state: Dict) -> Dict[str, float]:
        """
        모든 에이전트의 보상 계산
        Args:
            state: 현재 상태
            next_state: 다음 상태
        Returns:
            각 에이전트의 보상
        """
        rewards = {}
        
        # Catcher 보상 계산: R_t = ∑ĉ(I^t, J^t) + R^T̃_ca
        for catcher_id in [aid for aid in state['positions'].keys() if 'catcher' in aid]:
            # R^T̃_ca: 충돌 회피 보상
            collision_reward = self.calculate_collision_term(catcher_id, state)
            
            # ∑ĉ(I^t, J^t): Runner와의 상호작용 보상
            interaction_reward = self.calculate_interaction_term(catcher_id, state)
            
            rewards[catcher_id] = collision_reward + interaction_reward
            
        # Runner 보상 계산: R_r = -∑ĉ(I^t, J^t)
        for runner_id in [aid for aid in state['positions'].keys() if 'runner' in aid]:
            # Runner의 보상은 Catcher와의 상호작용 보상의 반대
            rewards[runner_id] = -self.calculate_runner_interaction_term(runner_id, state)
            
        return rewards

    def calculate_collision_term(self, catcher_id: str, state: Dict) -> float:
        """
        충돌 회피 항 계산 (R^T̃_ca)
        R^T̃_ca = -∑_{t=0}^T ∑_{I^t ∈ K̃^t, J^t ∈ (K̃^t ∪ O^t)} ĉ(I^t, J^t)
        """
        catcher_pos = state['positions'][catcher_id]
        collision_cost = 0.0
        
        # 다른 Catcher들과의 충돌 비용
        for other_id, other_pos in state['positions'].items():
            if 'catcher' in other_id and other_id != catcher_id:
                dist = np.linalg.norm(catcher_pos - other_pos)
                collision_cost += self.collision_cost_function(dist)
        
        # 장애물과의 충돌 비용
        if 'obstacles' in state:
            for obstacle_pos in state['obstacles']:
                dist = np.linalg.norm(catcher_pos - obstacle_pos)
                collision_cost += self.collision_cost_function(dist)
                
        return -collision_cost  # 음수 비용 = 보상

    def calculate_interaction_term(self, catcher_id: str, state: Dict) -> float:
        """
        Runner와의 상호작용 항 계산 (ĉ(I^t, J^t))
        """
        catcher_pos = state['positions'][catcher_id]
        interaction_reward = 0.0
        
        # 모든 Runner와의 상호작용
        for runner_id, runner_pos in state['positions'].items():
            if 'runner' in runner_id:
                dist = np.linalg.norm(catcher_pos - runner_pos)
                interaction_reward += self.interaction_reward_function(dist)
                
        return interaction_reward

    def calculate_runner_interaction_term(self, runner_id: str, state: Dict) -> float:
        """
        Runner의 Catcher와의 상호작용 항 계산
        """
        runner_pos = state['positions'][runner_id]
        interaction_cost = 0.0
        
        # 모든 Catcher와의 상호작용
        for catcher_id, catcher_pos in state['positions'].items():
            if 'catcher' in catcher_id:
                dist = np.linalg.norm(runner_pos - catcher_pos)
                interaction_cost += self.interaction_reward_function(dist)
                
        return interaction_cost

    def collision_cost_function(self, distance: float) -> float:
        """
        충돌 비용 함수 ĉ(I^t, J^t) for collision
        """
        if distance < self.collision_threshold:
            return 1.0
        return 0.0

    def interaction_reward_function(self, distance: float) -> float:
        """
        상호작용 보상 함수 ĉ(I^t, J^t) for capture
        """
        if distance < self.capture_threshold:
            return 1.0  # 포획 성공
        return 0.0  # 포획 실패

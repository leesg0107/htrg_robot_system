import numpy as np

class RewardCalculator:
    def __init__(self, config):
        """
        보상 계산기 초기화
        Args:
            config: 환경 설정 딕셔너리
        """
        self.capture_radius = config['capture_radius']
        self.collision_radius = config['collision_radius']
    
    def compute_rewards(self, agent_positions, captures, collisions):
        """
        보상 계산
        Args:
            agent_positions (dict): 각 에이전트의 위치
            captures (int): 현재까지의 포획 횟수
            collisions (int): 현재까지의 충돌 횟수
        Returns:
            dict: 각 에이전트의 보상
        """
        rewards = {}
        
        # 포획자와 도망자 위치 분리
        catcher_positions = {
            agent_id: pos 
            for agent_id, pos in agent_positions.items() 
            if 'catcher' in agent_id
        }
        runner_positions = {
            agent_id: pos 
            for agent_id, pos in agent_positions.items() 
            if 'runner' in agent_id
        }
        
        # 포획 보상 계산
        for runner_id, runner_pos in runner_positions.items():
            is_captured = False
            for catcher_id, catcher_pos in catcher_positions.items():
                distance = np.linalg.norm(runner_pos - catcher_pos)
                
                if distance < self.capture_radius:
                    is_captured = True
                    # 포획자 보상
                    rewards[catcher_id] = 1.0
                    # 다른 포획자들 보상
                    for other_catcher_id in catcher_positions.keys():
                        if other_catcher_id != catcher_id:
                            rewards[other_catcher_id] = 0.5
                    # 도망자 페널티
                    rewards[runner_id] = -1.0
                    break
            
            # 포획되지 않은 경우
            if not is_captured:
                # 도망자 보상
                rewards[runner_id] = 0.1
                # 포획자 페널티
                for catcher_id in catcher_positions.keys():
                    rewards[catcher_id] = -0.1
        
        # 충돌 페널티 계산
        for agent1_id, pos1 in agent_positions.items():
            for agent2_id, pos2 in agent_positions.items():
                if agent1_id < agent2_id:  # 중복 체크 방지
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance < self.collision_radius:
                        rewards[agent1_id] = rewards.get(agent1_id, 0.0) - 0.5
                        rewards[agent2_id] = rewards.get(agent2_id, 0.0) - 0.5
        
        # 기본 보상 설정
        for agent_id in agent_positions.keys():
            if agent_id not in rewards:
                rewards[agent_id] = 0.0
        
        return rewards

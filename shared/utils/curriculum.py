from typing import Dict, List
import numpy as np

class CurriculumLearning:
    def __init__(self):
        """
        Curriculum Learning 초기화
        - 3단계 커리큘럼: 기본 포획 -> 중급 포획 -> 고급 포획
        """
        self.stages = [
            {
                "name": "basic_catching",
                "n_catchers": 2,
                "n_runners": 1,
                "difficulty": "easy",
                "reward_weights": {
                    "capture": 1.0,
                    "collision": -0.5,
                    "distance": 0.3
                },
                "conditions": {
                    "min_success_rate": 0.7,
                    "min_episodes": 1000
                },
                "env_config": {
                    "grid_size": 10,
                    "max_steps": 100,
                    "obstacle_density": 0.0
                }
            },
            {
                "name": "intermediate_catching",
                "n_catchers": 2,
                "n_runners": 1,
                "difficulty": "medium",
                "reward_weights": {
                    "capture": 1.0,
                    "collision": -1.0,
                    "distance": 0.2
                },
                "conditions": {
                    "min_success_rate": 0.6,
                    "min_episodes": 1500
                },
                "env_config": {
                    "grid_size": 15,
                    "max_steps": 150,
                    "obstacle_density": 0.1
                }
            },
            {
                "name": "advanced_catching",
                "n_catchers": 3,
                "n_runners": 2,
                "difficulty": "hard",
                "reward_weights": {
                    "capture": 1.0,
                    "collision": -1.5,
                    "distance": 0.1
                },
                "conditions": {
                    "min_success_rate": 0.5,
                    "min_episodes": 2000
                },
                "env_config": {
                    "grid_size": 20,
                    "max_steps": 200,
                    "obstacle_density": 0.2
                }
            }
        ]
        self.current_stage = 0
        self.stage_history = []  # 스테이지 진행 기록

    def get_current_config(self) -> Dict:
        """현재 스테이지 설정 반환"""
        return self.stages[self.current_stage]

    def check_stage_completion(self, metrics: Dict) -> bool:
        """
        현재 스테이지 완료 여부 확인
        Args:
            metrics: 현재 성능 지표
                - success_rate: 포획 성공률
                - episodes: 진행된 에피소드 수
        """
        current = self.stages[self.current_stage]
        conditions = current["conditions"]
        
        # 모든 조건 충족 확인
        success_condition = metrics["success_rate"] >= conditions["min_success_rate"]
        episode_condition = metrics["episodes"] >= conditions["min_episodes"]
        
        if success_condition and episode_condition:
            # 스테이지 완료 기록
            self.stage_history.append({
                "stage": self.current_stage,
                "name": current["name"],
                "episodes": metrics["episodes"],
                "success_rate": metrics["success_rate"]
            })
            return True
            
        return False

    def advance_stage(self) -> bool:
        """
        다음 스테이지로 진행
        Returns:
            bool: 다음 스테이지 존재 여부
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def adjust_rewards(self, base_rewards: Dict) -> Dict:
        """
        현재 스테이지에 맞게 보상 조정
        Args:
            base_rewards: 기본 보상 값
        Returns:
            조정된 보상 값
        """
        weights = self.stages[self.current_stage]["reward_weights"]
        return {
            k: v * weights.get(k, 1.0) for k, v in base_rewards.items()
        }

    def get_stage_progress(self) -> Dict:
        """
        현재까지의 학습 진행 상황 반환
        """
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "stage_name": self.stages[self.current_stage]["name"],
            "stage_history": self.stage_history
        }

    def get_env_config(self) -> Dict:
        """현재 스테이지의 환경 설정 반환"""
        return self.stages[self.current_stage]["env_config"]

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
                "conditions": {
                    "min_success_rate": 0.6,
                    "min_episodes": 100,
                    "max_collision_rate": 0.3,
                    "min_avg_reward": -1.0
                },
                "config": {
                    "grid_size": 20,
                    "n_catchers": 2,
                    "n_runners": 1,
                    "obstacle_density": 0.0
                }
            },
            {
                "name": "intermediate_catching",
                "conditions": {
                    "min_success_rate": 0.7,
                    "min_episodes": 200,
                    "max_collision_rate": 0.2,
                    "min_avg_reward": 0.0
                },
                "config": {
                    "grid_size": 30,
                    "n_catchers": 2,
                    "n_runners": 1,
                    "obstacle_density": 0.1
                }
            }
        ]
        self.current_stage = 0
        self.episode_count = 0
        self.metrics_history = {
            "success_rates": [],
            "collision_rates": [],
            "average_rewards": []
        }

    def get_current_config(self) -> Dict:
        """현재 스테이지의 환경 설정 반환"""
        return self.stages[self.current_stage]["config"]

    def check_stage_completion(self, metrics: Dict) -> bool:
        """
        현재 스테이지 완료 여부 확인
        """
        self.episode_count += 1
        
        # 메트릭 히스토리 업데이트
        self.metrics_history["success_rates"].append(metrics["success_rate"])
        self.metrics_history["collision_rates"].append(metrics["collision_rate"])
        self.metrics_history["average_rewards"].append(metrics["average_reward"])
        
        # 최근 100 에피소드의 평균 계산
        window_size = min(100, len(self.metrics_history["success_rates"]))
        avg_success_rate = np.mean(self.metrics_history["success_rates"][-window_size:])
        avg_collision_rate = np.mean(self.metrics_history["collision_rates"][-window_size:])
        avg_reward = np.mean(self.metrics_history["average_rewards"][-window_size:])
        
        current = self.stages[self.current_stage]
        conditions = current["conditions"]
        
        # 조건 체크
        success_condition = avg_success_rate >= conditions["min_success_rate"]
        episode_condition = self.episode_count >= conditions["min_episodes"]
        collision_condition = avg_collision_rate <= conditions["max_collision_rate"]
        reward_condition = avg_reward >= conditions["min_avg_reward"]
        
        # 현재 성능 출력
        print(f"\nCurrent Performance (Stage {self.current_stage}):")
        print(f"Success Rate: {avg_success_rate:.3f} (target: {conditions['min_success_rate']})")
        print(f"Collision Rate: {avg_collision_rate:.3f} (target: {conditions['max_collision_rate']})")
        print(f"Average Reward: {avg_reward:.3f} (target: {conditions['min_avg_reward']})")
        print(f"Episodes: {self.episode_count} (target: {conditions['min_episodes']})")
        
        # 모든 조건이 만족되면 True 반환
        if success_condition and episode_condition and collision_condition and reward_condition:
            print(f"\nStage {current['name']} completed!")
            return True
            
        return False

    def advance_stage(self) -> None:
        """다음 스테이지로 진행"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"\nAdvancing to stage: {self.stages[self.current_stage]['name']}")
            # 새 스테이지 시작 시 메트릭 초기화
            self.episode_count = 0
            self.metrics_history = {
                "success_rates": [],
                "collision_rates": [],
                "average_rewards": []
            }
        else:
            print("\nAll stages completed!")

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

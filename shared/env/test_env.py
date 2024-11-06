import numpy as np
from simulation_env import SimulationEnv

def test_env():
    # 기본 설정
    config = {
        'n_catchers': 2,
        'n_runners': 1,
        'grid_size': 10,
        'max_steps': 100,
        'n_channels': 3,
        'collision_radius': 1.0,
        'capture_radius': 1.5,
        'agent_speed': 1.0
    }
    
    # 환경 생성
    env = SimulationEnv(config)
    
    # 환경 초기화
    obs = env.reset()
    
    # 간단한 랜덤 행동으로 몇 스텝 테스트
    for i in range(5):
        # 랜덤 행동 생성
        actions = {
            'catcher_0': np.random.uniform(-1, 1, 2),
            'catcher_1': np.random.uniform(-1, 1, 2),
            'runner_0': np.random.uniform(-1, 1, 2)
        }
        
        # 환경 스텝 실행
        obs, rewards, dones, info = env.step(actions)
        
        # 결과 출력
        print(f"\nStep {i+1}")
        print(f"Rewards: {rewards}")
        print(f"Positions:")
        print(f"- Catchers: {env.catcher_positions}")
        print(f"- Runners: {env.runner_positions}")

if __name__ == "__main__":
    test_env()
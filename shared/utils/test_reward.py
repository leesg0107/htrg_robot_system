import numpy as np
from reward import RewardCalculator

def test_reward_calculator():
    # 테스트 설정
    config = {
        'collision_threshold': 1.0,
        'capture_threshold': 2.0
    }
    
    reward_calc = RewardCalculator(config)
    
    # 테스트 케이스 1: 기본 상태
    print("\n테스트 1: 기본 상태")
    state = {
        'positions': {
            'catcher_0': np.array([0.0, 0.0]),
            'catcher_1': np.array([3.0, 3.0]),
            'runner_0': np.array([5.0, 5.0])
        },
        'obstacles': [
            np.array([1.0, 1.0])
        ]
    }
    next_state = {
        'positions': {
            'catcher_0': np.array([0.5, 0.5]),
            'catcher_1': np.array([3.5, 3.5]),
            'runner_0': np.array([5.0, 5.0])
        },
        'obstacles': [
            np.array([1.0, 1.0])
        ]
    }
    
    rewards = reward_calc.calculate_rewards(state, next_state)
    print("보상:")
    for agent_id, reward in rewards.items():
        print(f"{agent_id}: {reward}")
        
    # 테스트 케이스 2: 충돌 상황
    print("\n테스트 2: Catcher 간 충돌")
    state['positions']['catcher_0'] = np.array([3.0, 3.0])  # catcher_1과 같은 위치
    rewards = reward_calc.calculate_rewards(state, next_state)
    print("보상:")
    for agent_id, reward in rewards.items():
        print(f"{agent_id}: {reward}")
        
    # 테스트 케이스 3: 포획 상황
    print("\n테스트 3: Runner 포획")
    state['positions']['catcher_0'] = np.array([5.0, 5.0])  # runner_0과 같은 위치
    rewards = reward_calc.calculate_rewards(state, next_state)
    print("보상:")
    for agent_id, reward in rewards.items():
        print(f"{agent_id}: {reward}")
        
    # 테스트 케이스 4: 장애물 충돌
    print("\n테스트 4: 장애물 충돌")
    state['positions']['catcher_0'] = np.array([1.0, 1.0])  # 장애물과 같은 위치
    rewards = reward_calc.calculate_rewards(state, next_state)
    print("보상:")
    for agent_id, reward in rewards.items():
        print(f"{agent_id}: {reward}")

if __name__ == "__main__":
    test_reward_calculator()

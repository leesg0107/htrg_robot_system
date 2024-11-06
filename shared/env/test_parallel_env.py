import numpy as np
from parallel_env import ParallelEnv
import time

def test_parallel_env():
    # 환경 설정
    env_config = {
        'n_catchers': 2,
        'n_runners': 1,
        'grid_size': 10,
        'max_steps': 100,
        'n_channels': 3,
        'collision_radius': 1.0,
        'capture_radius': 1.5,
        'agent_speed': 1.0
    }
    
    # 성능 비교를 위한 시간 측정
    start_time = time.time()
    
    # 병렬 환경 생성 (4개의 환경)
    print("병렬 환경 초기화 중...")
    parallel_env = ParallelEnv(env_config, num_envs=4)
    
    try:
        # 모든 환경 초기화
        print("\n환경 리셋 중...")
        observations = parallel_env.reset()
        print(f"\n초기 상태 (4개 환경):")
        for i, obs in enumerate(observations):
            print(f"\n환경 {i}:")
            for agent_id, observation in obs.items():
                print(f"- {agent_id} 관찰 크기: {observation.shape}")
        
        # 몇 스텝 실행
        for step in range(3):
            print(f"\n스텝 {step + 1}")
            
            # 각 환경에 대한 랜덤 행동 생성
            actions_list = []
            for env_idx in range(4):
                actions = {
                    'catcher_0': np.random.uniform(-1, 1, 2),
                    'catcher_1': np.random.uniform(-1, 1, 2),
                    'runner_0': np.random.uniform(-1, 1, 2)
                }
                actions_list.append(actions)
            
            # 병렬 스텝 실행
            observations, rewards, dones, infos = parallel_env.step(actions_list)
            
            # 결과 출력
            for env_idx in range(4):
                print(f"\n환경 {env_idx}:")
                print(f"보상:")
                for agent_id, reward in rewards[env_idx].items():
                    print(f"- {agent_id}: {reward:.3f}")
                print(f"포획 수: {infos[env_idx]['captures']}")
                print(f"충돌 수: {infos[env_idx]['collisions']}")
                
                # 에피소드 종료 체크
                if dones[env_idx]['all_done']:
                    print(f"환경 {env_idx} 에피소드 종료!")
    
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
    
    finally:
        # 환경 정리
        print("\n환경 정리 중...")
        parallel_env.close()
        
        # 총 실행 시간 출력
        end_time = time.time()
        print(f"\n총 실행 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    test_parallel_env()

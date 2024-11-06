import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from catcher import Catcher
from runner import Runner
from env.simulation_env import SimulationEnv

def test_agents():
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
    
    # 에이전트 설정
    agent_config = {
        'grid_size': env_config['grid_size'],
        'capture_radius': env_config['capture_radius'],
        'collision_penalty': -5.0,
        'escape_reward': 1.0,
        'capture_penalty': -10.0
    }
    
    # 환경과 에이전트 초기화
    env = SimulationEnv(env_config)
    catchers = [Catcher(agent_config) for _ in range(env_config['n_catchers'])]
    runners = [Runner(agent_config) for _ in range(env_config['n_runners'])]
    
    print("테스트 시작...")
    
    # 몇 개의 에피소드 실행
    n_episodes = 3
    for episode in range(n_episodes):
        print(f"\n에피소드 {episode + 1}")
        
        # 환경 초기화
        observations = env.reset()
        episode_rewards = {
            'catchers': [0] * env_config['n_catchers'],
            'runners': [0] * env_config['n_runners']
        }
        
        # 에피소드 실행
        for step in range(env_config['max_steps']):
            # 행동 선택
            actions = {}
            
            # Catcher 행동
            for i in range(env_config['n_catchers']):
                catcher_id = f'catcher_{i}'
                actions[catcher_id] = catchers[i].act(observations[catcher_id])
            
            # Runner 행동
            for i in range(env_config['n_runners']):
                runner_id = f'runner_{i}'
                actions[runner_id] = runners[i].act(observations[runner_id])
            
            # 환경 스텝 실행
            next_observations, rewards, dones, info = env.step(actions)
            
            # 보상 누적
            for i in range(env_config['n_catchers']):
                episode_rewards['catchers'][i] += rewards[f'catcher_{i}']
            for i in range(env_config['n_runners']):
                episode_rewards['runners'][i] += rewards[f'runner_{i}']
            
            # 학습 데이터 수집 (실제 학습은 생략)
            experience = {
                'states': observations,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_observations,
                'dones': dones
            }
            
            # 상태 업데이트
            observations = next_observations
            
            # 에피소드 종료 체크
            if dones['all_done']:
                break
            
            # 매 10스텝마다 상태 출력
            if step % 10 == 0:
                print(f"\n스텝 {step}:")
                print(f"포획 수: {info['captures']}")
                print(f"충돌 수: {info['collisions']}")
                print("누적 보상:")
                for i in range(env_config['n_catchers']):
                    print(f"- Catcher {i}: {episode_rewards['catchers'][i]:.2f}")
                for i in range(env_config['n_runners']):
                    print(f"- Runner {i}: {episode_rewards['runners'][i]:.2f}")
        
        # 에피소드 결과 출력
        print(f"\n에피소드 {episode + 1} 종료:")
        print(f"총 스텝 수: {step + 1}")
        print("최종 누적 보상:")
        for i in range(env_config['n_catchers']):
            print(f"- Catcher {i}: {episode_rewards['catchers'][i]:.2f}")
        for i in range(env_config['n_runners']):
            print(f"- Runner {i}: {episode_rewards['runners'][i]:.2f}")
    
    env.close()
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_agents()

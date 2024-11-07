import argparse
from shared.env.simulation_env import SimulationEnv
from shared.models.mapoca import MAPOCA
from shared.utils.curriculum import CurriculumLearning
from shared.logger import Logger
import pygame
import torch
import numpy as np 
import os
import matplotlib.pyplot as plt
from collections import deque

class MetricLogger:
    def __init__(self, window_size=100):
        # 메트릭 키 정의
        self.metric_keys = ['reward', 'success_rate', 'collision_rate', 'avg_catch_time', 'stage']
        
        # 저장소 초기화
        self.episodes = []
        self.metrics = {key: [] for key in self.metric_keys}
        
        # 이동 평균을 위한 윈도우 (stage 제외)
        self.windows = {
            key: deque(maxlen=window_size) 
            for key in self.metric_keys if key != 'stage'
        }
        
        # 평가 메트릭
        self.eval_episodes = []
        self.eval_metrics = {
            'eval_success_rate': [],
            'eval_catch_time': [],
            'eval_reward': [],
            'eval_collision_rate': []
        }
    
    def log_episode(self, episode, metrics_dict):
        """
        에피소드 메트릭 기록
        Args:
            episode (int): 현재 에피소드 번호
            metrics_dict (dict): 기록할 메트릭 딕셔너리
        """
        try:
            # 에피소드 번호 저장
            self.episodes.append(episode)
            
            # 각 메트릭 처리
            for key in self.metric_keys:
                if key not in metrics_dict:
                    print(f"Warning: Missing metric '{key}' in metrics_dict")
                    continue
                    
                value = metrics_dict[key]
                
                # stage는 이동 평균을 계산하지 않음
                if key == 'stage':
                    self.metrics[key].append(value)
                else:
                    # 윈도우에 추가하고 평균 계산
                    self.windows[key].append(value)
                    self.metrics[key].append(np.mean(self.windows[key]))
                    
        except Exception as e:
            print(f"Error in log_episode: {str(e)}")
            raise
    
    def plot_metrics(self, save_path='plots'):
        """메트릭 시각화 및 저장"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 1. 보상 그래프
            self._plot_single_metric(
                'reward', 
                'Training Rewards',
                'Reward',
                f'{save_path}/rewards.png'
            )
            
            # 2. 성공률과 충돌률
            self._plot_dual_metrics(
                'success_rate',
                'collision_rate',
                'Success and Collision Rates',
                'Rate',
                f'{save_path}/rates.png'
            )
            
            # 3. 평균 포획 시간
            self._plot_single_metric(
                'avg_catch_time',
                'Episode Length',
                'Steps',
                f'{save_path}/steps.png'
            )
            
            # 4. 커리큘럼 스테이지
            self._plot_single_metric(
                'stage',
                'Curriculum Learning Progress',
                'Stage',
                f'{save_path}/curriculum.png'
            )
            
        except Exception as e:
            print(f"Error in plot_metrics: {str(e)}")
            raise
    
    def _plot_single_metric(self, metric_key, title, ylabel, save_path):
        """단일 메트릭 그래프 생성"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.episodes, self.metrics[metric_key], label=title)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    def _plot_dual_metrics(self, key1, key2, title, ylabel, save_path):
        """두 메트릭을 함께 표시하는 그래프 생성"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.episodes, self.metrics[key1], label=key1.replace('_', ' ').title())
        plt.plot(self.episodes, self.metrics[key2], label=key2.replace('_', ' ').title())
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    def log_evaluation(self, episode, eval_metrics_dict):
        """
        평가 메트릭 기록
        Args:
            episode (int): 현재 에피소드 번호
            eval_metrics_dict (dict): 평가 결과 메트릭
        """
        try:
            self.eval_episodes.append(episode)
            
            # 평가 메트릭 저장
            self.eval_metrics['eval_success_rate'].append(eval_metrics_dict['success_rate'])
            self.eval_metrics['eval_catch_time'].append(eval_metrics_dict['avg_catch_time'])
            self.eval_metrics['eval_reward'].append(eval_metrics_dict['reward'])
            self.eval_metrics['eval_collision_rate'].append(eval_metrics_dict['collision_rate'])
            
            # 평가 결과 그래프 업데이트
            self.plot_eval_metrics()
            
        except Exception as e:
            print(f"Error in log_evaluation: {str(e)}")
            raise
    
    def plot_eval_metrics(self, save_path='plots'):
        """평가 메트릭 시각화"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 1. 평가 성공률
            self._plot_single_metric_eval(
                'eval_success_rate',
                'Evaluation Success Rate',
                'Success Rate',
                f'{save_path}/eval_success_rate.png'
            )
            
            # 2. 평가 포획 시간
            self._plot_single_metric_eval(
                'eval_catch_time',
                'Evaluation Catch Time',
                'Steps',
                f'{save_path}/eval_catch_time.png'
            )
            
            # 3. 평가 보상
            self._plot_single_metric_eval(
                'eval_reward',
                'Evaluation Reward',
                'Reward',
                f'{save_path}/eval_reward.png'
            )
            
            # 4. 평가 충돌률
            self._plot_single_metric_eval(
                'eval_collision_rate',
                'Evaluation Collision Rate',
                'Collision Rate',
                f'{save_path}/eval_collision_rate.png'
            )
            
        except Exception as e:
            print(f"Error in plot_eval_metrics: {str(e)}")
            raise
    
    def _plot_single_metric_eval(self, metric_key, title, ylabel, save_path):
        """평가 메트릭 그래프 생성"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_episodes, self.eval_metrics[metric_key], 
                label=title, marker='o')
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(args.log_dir)
    
    # 체크포인트 디���리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 환경 설정
    env_config = {
        'grid_size': 20,           # 격자 크기
        'n_catchers': 2,           # 포획자 수
        'n_runners': 1,            # 도망자 수
        'max_steps': 200,          # 최대 스텝 수
        'capture_radius': 1.0,     # 포획 반경
        'collision_radius': 0.5,    # 충돌 반경
        'render_mode': 'human',    # 렌더링 모드
        'fps': 30,                 # 프레임률
    }
    
    # 환경 초기화
    env = SimulationEnv(env_config)
    
    # 총 에이전트 수 계산
    n_agents = env_config['n_catchers'] + env_config['n_runners']
    
    # 상태 차원 계산 (observation_space가 3D array인 경우)
    obs_shape = env.observation_space.shape
    state_dim = obs_shape[0] * obs_shape[1] * obs_shape[2] if len(obs_shape) == 3 else obs_shape[0]
    
    # MAPOCA 설정
    model_config = {
        'state_dim': state_dim,                        # 상태 차원 (flatten된 크기)
        'action_dim': env.action_space.shape[0],       # 행동 차원
        'hidden_dim': 256,                             # 은닉층 차원
        'action_std': 0.1,                             # 행동 표준편차
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # 에이전트 설정
        'n_agents': n_agents,                          # 총 에이전트 수
        'n_catchers': env_config['n_catchers'],        # 포획자 수
        'n_runners': env_config['n_runners'],          # 도망자 수
        
        # 학습 설정
        'buffer_size': 10000,                          # 리플레이 버퍼 크기
        'batch_size': 64,                              # 배치 크기
        'learning_rate': 3e-4,                         # 학습률
        'gamma': 0.99,                                 # 할인 계수
        'tau': 0.005,                                  # 타겟 네트워크 업데이트 비율
        'update_interval': 100,                        # 업데이트 간격
        'clip_param': 0.2,                             # PPO 클리핑 파라미터
        'max_grad_norm': 0.5,                          # 그래디언트 클리핑
        'value_loss_coef': 0.5,                        # 가치 손실 계수
        'entropy_coef': 0.01,                          # 엔트로피 계수
        'n_epochs': 10,                                # PPO 에포크 수
        'gae_lambda': 0.95,                            # GAE 람다
        
        # 관찰/행동 공간 설정
        'observation_space': env.observation_space,     # 관찰 공간
        'action_space': env.action_space,              # 행동 공간
    }
    
    # 학습 설정
    train_config = {
        'num_episodes': 1000,      # 총 에피소드 수
        'eval_interval': 10,       # 평가 간격
        'save_interval': 100,      # 체크포인트 저장 간격
        'log_interval': 1,         # 로깅 간격
        'render_interval': 10,     # 렌더링 간격
    }
    
    print("\nEnvironment Configuration:")
    print(f"Number of Catchers: {env_config['n_catchers']}")
    print(f"Number of Runners: {env_config['n_runners']}")
    print(f"Total Agents: {n_agents}")
    print(f"Original Observation Shape: {obs_shape}")
    print(f"Flattened State Dimension: {state_dim}")
    print(f"Action Dimension: {env.action_space.shape[0]}")
    
    # MAPOCA 초기화
    mapoca = MAPOCA(model_config)
    
    # 메트릭 로거 초기화
    logger = MetricLogger()
    
    # 커리큘럼 학습 초기화
    curriculum = CurriculumLearning()
    
    # 체크포인트 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    
    # 에이전트 ID 매핑
    agent_id_map = {
        'catcher_0': 0,
        'catcher_1': 1,
        'runner_0': 2
    }
    
    # 학습 루프
    for episode in range(train_config['num_episodes']):
        state_dict = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_captures = 0
        episode_collisions = 0
        done = False
        
        while not done:
            # 행동 선택
            actions_dict = {}
            for agent_id, state in state_dict.items():
                state_tensor = torch.FloatTensor(state).to(model_config['device'])
                action = mapoca.select_action(state_tensor, agent_id)
                actions_dict[agent_id] = action.cpu().numpy()
            
            # 환경 스텝
            next_states_dict, rewards_dict, dones_dict, info = env.step(actions_dict)
            
            # 경험 저장
            for agent_id in state_dict.keys():
                state = state_dict[agent_id]
                action = actions_dict[agent_id]
                reward = rewards_dict[agent_id]
                next_state = next_states_dict[agent_id]
                done = dones_dict[agent_id]
                
                # 숫자 ID로 변환하여 저장
                numeric_id = agent_id_map[agent_id]
                mapoca.store_transition(
                    numeric_id,
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            
            # 상태 업데이트
            state_dict = next_states_dict
            episode_reward += sum(rewards_dict.values())
            episode_steps += 1
            episode_captures += info.get('captures', 0)
            episode_collisions += info.get('collisions', 0)
            
            # 모델 업데이트
            if mapoca.is_ready_to_update():
                losses = mapoca.update()
                if losses and episode % train_config['log_interval'] == 0:
                    print("\nTraining Metrics:")
                    for key, value in losses.items():
                        print(f"{key}: {value:.4f}")
            
            # 현재 상태 출력
            if episode % train_config['log_interval'] == 0:
                print(f"\nStep {episode_steps}:")
                print(f"Actions: {actions_dict}")
                print(f"Rewards: {rewards_dict}")
            
            if done:
                break
        
        # 에피소드 종료 시 메트릭 기록
        metrics = {
            "reward": episode_reward,
            "success_rate": float(episode_captures > 0),
            "collision_rate": float(episode_collisions) / max(episode_steps, 1),
            "avg_catch_time": episode_steps,
            "stage": curriculum.current_stage
        }
        
        logger.log_episode(episode, metrics)
        
        # 주기적으로 그래프 업데이트
        if episode % train_config['render_interval'] == 0:
            logger.plot_metrics()
        
        # 주기적으로 모델 평가
        if episode % train_config['eval_interval'] == 0:
            eval_metrics = evaluate(env, mapoca, num_episodes=5)
            logger.log_evaluation(episode, eval_metrics)
            
            print("\nEvaluation Results:")
            print(f"Success Rate: {eval_metrics['success_rate']:.3f}")
            print(f"Average Catch Time: {eval_metrics['avg_catch_time']:.1f}")
            print(f"Average Reward: {eval_metrics['reward']:.3f}")
            print(f"Collision Rate: {eval_metrics['collision_rate']:.3f}")
        
        # 주기적으로 체크포인트 저장
        if episode % train_config['save_interval'] == 0:
            mapoca.save_checkpoint(f'checkpoints/episode_{episode}.pt')

def process_states(states_dict, n_catchers, n_runners):
    """상태 딕셔너리를 텐서로 변환"""
    states_list = []
    
    # Catchers 상태
    for i in range(n_catchers):
        state = states_dict[f'catcher_{i}']
        states_list.append(torch.FloatTensor(state.flatten()))
    
    # Runners 상태
    for i in range(n_runners):
        state = states_dict[f'runner_{i}']
        states_list.append(torch.FloatTensor(state.flatten()))
    
    return torch.stack(states_list)

def process_actions(actions_tensor, n_catchers, n_runners):
    """행동 텐서를 딕셔너리로 변환"""
    actions_dict = {}
    
    # Catchers 행동
    for i in range(n_catchers):
        actions_dict[f'catcher_{i}'] = actions_tensor[i].numpy()
    
    # Runners 행동
    for i in range(n_runners):
        actions_dict[f'runner_{i}'] = actions_tensor[n_catchers + i].numpy()
    
    return actions_dict

def process_rewards(rewards_dict, n_catchers, n_runners):
    """보상 딕셔너리를 텐서로 변환"""
    rewards_list = []
    
    # Catchers 보상
    for i in range(n_catchers):
        rewards_list.append(rewards_dict[f'catcher_{i}'])
    
    # Runners 보상
    for i in range(n_runners):
        rewards_list.append(rewards_dict[f'runner_{i}'])
    
    return torch.FloatTensor(rewards_list)

def evaluate(env, model, num_episodes=5):
    """
    모델 성능 평가
    Args:
        env: 환경
        model: 평가할 모델
        num_episodes: 평가할 에피소드 수
    Returns:
        dict: 평가 메트릭
    """
    # 평가 모드로 설정
    model.train(False)
    
    total_reward = 0
    total_steps = 0
    total_captures = 0
    total_collisions = 0
    
    try:
        for _ in range(num_episodes):
            state_dict = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # 행동 선택
                actions_dict = {}
                for agent_id, state in state_dict.items():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = model.select_action(state_tensor, agent_id)
                    actions_dict[agent_id] = action.squeeze().numpy()
                
                # 환경 스텝
                next_states_dict, rewards_dict, dones_dict, info = env.step(actions_dict)
                
                # 메트릭 업데이트
                episode_reward += sum(rewards_dict.values())
                episode_steps += 1
                
                # 상태 업데이트
                state_dict = next_states_dict
                done = all(dones_dict.values())
            
            # 에피소드 메트릭 누적
            total_reward += episode_reward
            total_steps += episode_steps
            total_captures += info.get('captures', 0)
            total_collisions += info.get('collisions', 0)
    
    finally:
        # 학습 모드로 복귀
        model.train(True)
    
    # 평균 메트릭 계산
    num_episodes = max(num_episodes, 1)  # 0으로 나누기 방지
    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    success_rate = total_captures / num_episodes
    collision_rate = total_collisions / total_steps if total_steps > 0 else 0
    
    return {
        'reward': avg_reward,
        'avg_catch_time': avg_steps,
        'success_rate': success_rate,
        'collision_rate': collision_rate
    }

if __name__ == "__main__":
    main()

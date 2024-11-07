import torch
import numpy as np
from models.mapoca import MAPOCA
from utils.curriculum import CurriculumLearning

def test_mapoca_basic():
    """MA-POCA 기본 기능 테스트"""
    print("\n=== MA-POCA 기본 기능 테스트 ===")
    
    # 설정 보강
    config = {
        'state_dim': 10,
        'action_dim': 2,
        'n_agents': 3,
        'device': 'cpu',
        'batch_size': 4,
        'gamma': 0.99,
        'lambda': 0.95,
        'epsilon': 0.2,
        'policy_lr': 3e-4,
        'value_lr': 1e-3
    }
    
    # 메트릭 추가
    metrics = {
        'episode_rewards': [],
        'success_rate': 0.0,
        'collision_rate': 0.0
    }
    
    mapoca = MAPOCA(config)
    
    # 테스트 배치에 더 현실적인 데이터 추가
    batch = {
        'states': torch.randn(4, 3, 10),
        'actions': torch.randn(4, 3, 2).clamp(-1, 1),  # 행동 범위 제한
        'old_probs': torch.randn(4, 3, 2),
        'rewards': torch.randn(4, 3) * 0.1,  # 보상 스케일 조정
        'dones': torch.zeros(4, 3, dtype=torch.bool)
    }
    
    # 성능 지표 출력 추가
    state = torch.randn(1, 10)
    action = mapoca.select_action(state)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    losses = mapoca.update(batch)
    print("\nTraining losses:")
    for key, value in losses.items():
        print(f"{key}: {value:.3f}")

def test_mapoca_with_curriculum():
    """MA-POCA와 Curriculum Learning 통합 테스트"""
    print("\n=== MA-POCA + Curriculum Learning 통합 테스트 ===")
    
    # Curriculum 설정
    curriculum = CurriculumLearning()
    current_stage = curriculum.get_current_config()
    print(f"\n현재 스테이지: {current_stage['name']}")
    
    # MA-POCA 설정 업데이트
    config = {
        'state_dim': 10,
        'action_dim': 2,
        'n_agents': current_stage['n_catchers'] + current_stage['n_runners'],
        'device': 'cpu',
        'batch_size': 4
    }
    
    mapoca = MAPOCA(config)
    
    # 간단한 학습 루프 시뮬레이션
    n_episodes = 10 
    for episode in range(n_episodes):
        print(f"\n에피소드 {episode + 1}")
        
        # 가상의 에피소드 데이터
        batch = {
            'states': torch.randn(4, config['n_agents'], 10),
            'actions': torch.randn(4, config['n_agents'], 2),
            'old_probs': torch.randn(4, config['n_agents'], 2),
            'rewards': torch.randn(4, config['n_agents']),
            'dones': torch.zeros(4, config['n_agents'], dtype=torch.bool)
        }
        
        # MA-POCA 업데이트
        losses = mapoca.update(batch)
        print("Losses:", {k: f"{v:.3f}" for k, v in losses.items()})
        
        # 가상의 메트릭스
        metrics = {
            'success_rate': 0.7 + episode * 0.05,  # 점진적 성능 향상 시뮬레이션
            'episodes': episode + 1
        }
        
        # 스테이지 완료 체크
        if curriculum.check_stage_completion(metrics):
            if curriculum.advance_stage():
                print(f"\n스테이지 진행! 새로운 스테이지: {curriculum.get_current_config()['name']}")
                # MA-POCA 설정 업데이트
                config['n_agents'] = (curriculum.get_current_config()['n_catchers'] + 
                                    curriculum.get_current_config()['n_runners'])
            else:
                print("\n모든 스테이지 완료!")
                break

if __name__ == "__main__":
    test_mapoca_basic()
    test_mapoca_with_curriculum()

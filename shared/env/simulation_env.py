import numpy as np
from typing import Dict, Tuple, List
from shared.env.base_env import BaseEnv
import pygame
from shared.utils.reward import RewardCalculator

class SimulationEnv(BaseEnv):
    """
    Simulation environment for the catch-runner scenario.
    Implements the actual game logic and physics.
    """
    def __init__(self, config: Dict):
        """
        Initialize simulation environment
        
        Args:
            config: Dictionary containing:
                - grid_size: Size of the grid (int)
                - max_steps: Maximum steps per episode (int)
                - render_mode: Rendering mode (str)
                - n_catchers: Number of catcher agents (int)
                - n_runners: Number of runner agents (int)
                - collision_radius: Radius for collision detection (float)
                - capture_radius: Radius for capture detection (float)
                - agent_speed: Maximum speed of agents (float)
        """
        # 환경 설정
        self.grid_size = config['grid_size']
        self.n_catchers = config['n_catchers']
        self.n_runners = config['n_runners']
        self.max_steps = config['max_steps']
        self.capture_radius = config['capture_radius']
        self.collision_radius = config['collision_radius']
        self.render_mode = config['render_mode']
        self.fps = config['fps']
        
        # 보상 계산기 초기화
        self.reward_calculator = RewardCalculator(config)
        
        # 에이전트 속성
        self.agent_speed = 0.5  # 에이전트 이동 속도
        self.agent_positions = {}  # 에이전트 위치 저장
        
        # 관찰/행동 공간 정의
        self.observation_space = np.zeros((self.grid_size, self.grid_size, 3))
        self.action_space = np.zeros(2)  # [dx, dy]
        
        # 게임 상태
        self.steps = 0
        self.captures = 0
        self.collisions = 0
        
        # 초기 위치 설정
        self.reset()
        
        # PyGame 초기화
        if self.render_mode == 'human':
            pygame.init()
            self.screen_size = 800
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Catch-Runner Simulation")
            
            # 시각화 관련 설정
            self.agent_radius = 20
            self.colors = {
                'catcher': (255, 0, 0),    # 빨강
                'runner': (0, 0, 255),     # 파랑
                'obstacle': (128, 128, 128) # 회색
            }
    
    def reset(self) -> Dict:
        """환경 초기화"""
        self.steps = 0
        self.captures = 0
        self.collisions = 0
        
        # 에이전트 위치 초기화 (랜덤하게)
        self.agent_positions = {}
        
        # 포획자 초기 위치 (그리드의 상단)
        for i in range(self.n_catchers):
            self.agent_positions[f'catcher_{i}'] = np.array([
                np.random.uniform(0, self.grid_size),
                np.random.uniform(self.grid_size * 0.7, self.grid_size)
            ])
        
        # 도망자 초기 위치 (그리드의 하단)
        for i in range(self.n_runners):
            self.agent_positions[f'runner_{i}'] = np.array([
                np.random.uniform(0, self.grid_size),
                np.random.uniform(0, self.grid_size * 0.3)
            ])
        
        return self._get_observations()
    
    def _get_observations(self):
        """각 에이전트의 관찰 상태 반환"""
        observations = {}
        
        # 모든 에이전트에 대해 관찰 생성
        for agent_id in self.agent_positions.keys():
            # 현재 에이전트의 위치
            current_pos = self.agent_positions[agent_id]
            
            # 관찰 배열 초기화
            obs = np.zeros((self.grid_size, self.grid_size, 3))
            
            # 모든 에이전트의 위치를 관찰에 표시
            for other_id, other_pos in self.agent_positions.items():
                x, y = other_pos.astype(int)
                x = np.clip(x, 0, self.grid_size-1)
                y = np.clip(y, 0, self.grid_size-1)
                
                if 'catcher' in other_id:
                    obs[x, y, 0] = 1  # 포획자는 빨간색 채널
                else:
                    obs[x, y, 1] = 1  # 도망자는 녹색 채널
            
            observations[agent_id] = obs
        
        return observations
    
    def step(self, actions):
        """
        환경 진행
        Args:
            actions (dict): 각 에이전트의 행동
        Returns:
            tuple: (next_states, rewards, dones, info)
        """
        # 위치 업데이트
        self._update_positions(actions)
        
        # 보상 계산
        rewards = self.reward_calculator.compute_rewards(
            self.agent_positions,
            self.captures,
            self.collisions
        )
        
        # 상태 업데이트
        self.steps += 1
        
        # 종료 조건 확인
        done = self.steps >= self.max_steps
        dones = {agent_id: done for agent_id in self.agent_positions.keys()}
        
        # 다음 상태 관찰
        next_states = self._get_observations()
        
        # 추가 정보
        info = {
            'captures': self.captures,
            'collisions': self.collisions
        }
        
        return next_states, rewards, dones, info
    
    def get_observation(self, agent_id: str) -> np.ndarray:
        """Generate observation for specific agent."""
        # Create grid-like sensor observation
        obs = np.zeros((self.grid_size, self.grid_size, 3))  # 3 channels: catchers, runners, obstacles
        
        # Fill observation channels
        if 'catcher' in agent_id:
            agent_pos = self.catcher_positions[agent_id]
        else:
            agent_pos = self.runner_positions[agent_id]
            
        # Add agents and obstacles to observation (with partial observability)
        self._add_visible_agents_to_obs(obs, agent_pos)
        self._add_obstacles_to_obs(obs)
        
        return obs
    
    def _reset_positions(self):
        """Reset agent positions randomly while avoiding collisions."""
        # Initialize empty positions
        self.catcher_positions = {}
        self.runner_positions = {}
        
        # Place catchers
        for i in range(self.n_catchers):
            while True:
                pos = np.random.uniform(0, self.grid_size, 2)
                if self._is_valid_position(pos):
                    self.catcher_positions[f'catcher_{i}'] = pos
                    break
        
        # Place runners
        for i in range(self.n_runners):
            while True:
                pos = np.random.uniform(0, self.grid_size, 2)
                if self._is_valid_position(pos):
                    self.runner_positions[f'runner_{i}'] = pos
                    break
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """
        Check if position is valid (no collisions with other agents or obstacles).
        """
        # Check boundaries
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
            
        # Check collision with obstacles
        for obs_pos in self.obstacles:
            if np.linalg.norm(pos - obs_pos) < self.collision_radius:
                return False
                
        # Check collision with other agents
        for catcher_pos in self.catcher_positions.values():
            if np.linalg.norm(pos - catcher_pos) < self.collision_radius:
                return False
                
        for runner_pos in self.runner_positions.values():
            if np.linalg.norm(pos - runner_pos) < self.collision_radius:
                return False
                
        return True
    
    def _update_positions(self, actions):
        """에이전트 위치 업데이트"""
        print("\nPosition updates:")
        for agent_id, action in actions.items():
            current_pos = self.agent_positions[agent_id]
            
            # 행동을 이동량으로 변환
            dx = float(action[0]) * self.agent_speed
            dy = float(action[1]) * self.agent_speed
            
            # 새 위치 계산 (경계 체크 포함)
            new_x = np.clip(current_pos[0] + dx, 0, self.grid_size)
            new_y = np.clip(current_pos[1] + dy, 0, self.grid_size)
            new_pos = np.array([new_x, new_y])
            
            print(f"{agent_id}:")
            print(f"  Current pos: {current_pos}")
            print(f"  Action: {action}")
            print(f"  Delta: ({dx:.3f}, {dy:.3f})")
            print(f"  New pos: {new_pos}")
            
            self.agent_positions[agent_id] = new_pos
    
    def _add_visible_agents_to_obs(self, obs: np.ndarray, agent_pos: np.ndarray):
        """Add visible agents to observation grid."""
        view_range = self.grid_size // 2
        
        # Add catchers to channel 0
        for pos in self.catcher_positions.values():
            if np.linalg.norm(pos - agent_pos) < view_range:
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    obs[grid_x, grid_y, 0] = 1
                    
        # Add runners to channel 1
        for pos in self.runner_positions.values():
            if np.linalg.norm(pos - agent_pos) < view_range:
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    obs[grid_x, grid_y, 1] = 1
    
    def _add_obstacles_to_obs(self, obs: np.ndarray):
        """Add obstacles to observation grid."""
        # Add obstacles to channel 2
        for pos in self.obstacles:
            grid_x = int(pos[0])
            grid_y = int(pos[1])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                obs[grid_x, grid_y, 2] = 1
    
    def render(self):
        """환경 상태를 시각화"""
        # 화면 초기화
        self.screen.fill((255, 255, 255))
        
        # 격자 그리기
        cell_size = self.screen_size / self.grid_size
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (0, i * cell_size), 
                           (self.screen_size, i * cell_size))
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (i * cell_size, 0), 
                           (i * cell_size, self.screen_size))
        
        # 장애물 그리기
        for obs_pos in self.obstacles:
            screen_pos = self._grid_to_screen(obs_pos)
            pygame.draw.circle(self.screen, self.colors['obstacle'], 
                             screen_pos, self.agent_radius)
        
        # Catcher 그리기
        for pos in self.catcher_positions.values():
            screen_pos = self._grid_to_screen(pos)
            pygame.draw.circle(self.screen, self.colors['catcher'], 
                             screen_pos, self.agent_radius)
        
        # Runner 그리기
        for pos in self.runner_positions.values():
            screen_pos = self._grid_to_screen(pos)
            pygame.draw.circle(self.screen, self.colors['runner'], 
                             screen_pos, self.agent_radius)
        
        # 화면 업데이트
        pygame.display.flip()
        
    def _grid_to_screen(self, pos):
        """격자 좌표를 화면 좌표로 변환"""
        x = pos[0] * self.screen_size / self.grid_size
        y = pos[1] * self.screen_size / self.grid_size
        return (int(x), int(y))
        
    def close(self):
        """환경 종료"""
        pygame.quit()
        super().close()
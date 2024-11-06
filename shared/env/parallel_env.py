import numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple
from simulation_env import SimulationEnv

class ParallelEnv:
    """
    Multiple environments running in parallel for faster training
    """
    def __init__(self, env_config: Dict, num_envs: int = 4):
        """
        Initialize parallel environments
        
        Args:
            env_config: Configuration for each environment
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.env_config = env_config
        # 각 환경을 인스턴스 변수로 유지
        self.envs = [SimulationEnv(env_config) for _ in range(num_envs)]
        self.pool = mp.Pool(processes=num_envs)
        
    def reset(self) -> List[Dict]:
        """병렬로 모든 환경 초기화"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions_list: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """병렬로 모든 환경 스텝 실행"""
        results = []
        for env_idx, (env, actions) in enumerate(zip(self.envs, actions_list)):
            result = env.step(actions)
            results.append(result)
        
        # 결과 언패킹
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)
    
    def close(self):
        """환경과 프로세스 풀 종료"""
        for env in self.envs:
            env.close()
        self.pool.close()
        self.pool.join()

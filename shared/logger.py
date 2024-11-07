from collections import defaultdict
import json
import os
import time
from typing import Dict, Any

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # 로그 파일 초기화
        self.log_file = os.path.join(log_dir, f"training_log_{int(self.start_time)}.txt")
        
    def log_training(self, episode: int, losses: Dict[str, float]):
        """학습 중 발생하는 loss를 기록"""
        log_str = f"Episode {episode} - Losses: {json.dumps(losses)}"
        self._write_log(log_str)
        
        for key, value in losses.items():
            self.metrics[f"loss_{key}"].append(value)
    
    def log_episode(self, episode: int, info: Dict[str, Any]):
        """에피소드 정보를 기록"""
        log_str = f"Episode {episode} - Info: {json.dumps(info)}"
        self._write_log(log_str)
        
        for key, value in info.items():
            self.metrics[key].append(value)
    
    def log_evaluation(self, episode: int, metrics: Dict[str, float]):
        """평가 결과를 기록"""
        log_str = f"Evaluation at episode {episode} - Metrics: {json.dumps(metrics)}"
        self._write_log(log_str)
        
        for key, value in metrics.items():
            self.metrics[f"eval_{key}"].append(value)
    
    def get_metrics(self) -> Dict[str, list]:
        """현재까지의 모든 메트릭 반환"""
        return dict(self.metrics)
    
    def _write_log(self, log_str: str):
        """로그를 파일에 기록하고 콘솔에 출력"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        full_log = f"[{timestamp}] {log_str}"
        
        print(full_log)
        with open(self.log_file, 'a') as f:
            f.write(full_log + '\n')

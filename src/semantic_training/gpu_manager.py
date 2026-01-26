import torch
import time
import logging
from parameter import *

class GPUMemoryManager:
    def __init__(self, enable_balancing=ENABLE_GPU_BALANCING, memory_threshold=GPU_MEMORY_THRESHOLD):
        self.enable_balancing = enable_balancing
        self.memory_threshold = memory_threshold
        self.num_gpus = torch.cuda.device_count()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('GPUMonitor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_memory_status(self):
        """Returns dict of memory usage per GPU in MB"""
        status = {}
        for i in range(self.num_gpus):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**2
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            status[i] = {
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'usage_percent': allocated / total
            }
        return status

    def log_status(self):
        """Logs current memory status"""
        status = self.get_memory_status()
        msg = "GPU Memory Status:\n"
        for i, s in status.items():
            msg += f"  GPU {i}: Alloc {s['allocated']:.1f}MB / {s['total']:.1f}MB ({s['usage_percent']*100:.1f}%)\n"
        self.logger.info(msg)
        return status

    def check_and_balance(self):
        """Checks for imbalance or high usage and suggests actions"""
        if not self.enable_balancing:
            return

        status = self.get_memory_status()
        
        # Check Thresholds
        for i, s in status.items():
            if s['usage_percent'] > self.memory_threshold:
                self.logger.warning(f"GPU {i} memory usage critical: {s['usage_percent']*100:.1f}%! Suggesting cache clear.")
                self.safe_empty_cache()

        # Check Imbalance (Simple Heuristic)
        if self.num_gpus >= 2:
            usage0 = status[0]['usage_percent']
            usage1 = status[1]['usage_percent']
            if abs(usage0 - usage1) > GPU_IMBALANCE_THRESHOLD:
                self.logger.info(f"Detected Memory Imbalance: GPU0 {usage0*100:.1f}% vs GPU1 {usage1*100:.1f}%.")
                # In a dynamic system we would migrate tasks here.
                # For this static implementation, we just log.

    def safe_empty_cache(self):
        """Safely empties cache"""
        torch.cuda.empty_cache()
        self.logger.info("Executed torch.cuda.empty_cache()")

    def suggest_device_map(self):
        """Returns a suggested device map for networks based on static load balancing"""
        # Static Strategy: 
        # If 2 GPUs: Split Policy/Q1 (GPU0) and Q2 (GPU1)
        if self.num_gpus >= 2 and self.enable_balancing:
            return {
                'policy': 'cuda:0',
                'q1': 'cuda:0',
                'q2': 'cuda:1'
            }
        else:
            return {
                'policy': 'cuda:0' if USE_GPU else 'cpu',
                'q1': 'cuda:0' if USE_GPU else 'cpu',
                'q2': 'cuda:0' if USE_GPU else 'cpu'
            }

    @staticmethod
    def safe_execution(func):
        """Decorator/Wrapper for OOM safety"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("OOM Detected! Clearing cache and skipping step.")
                    torch.cuda.empty_cache()
                    return None
                else:
                    raise e
        return wrapper

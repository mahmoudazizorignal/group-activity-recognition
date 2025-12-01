import math
from models.lr.LRInterface import LRInterface

class CosineLR(LRInterface):
    
    def __init__(self, max_steps, warmup_steps, max_lr, min_lr):
        
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, it: int) -> float:
        
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        
        if it > self.max_steps:
            return self.min_lr
        
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

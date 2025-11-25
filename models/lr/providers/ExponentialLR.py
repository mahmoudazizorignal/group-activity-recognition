from models.lr.LRInterface import LRInterface

class ExponentialLR(LRInterface):

    def __init__(self, initial_lr: float, beta: float):
        assert 0 <= beta <= 1.0
        self.lr = initial_lr
        self.beta = beta
    
    def get_lr(self, it=None) -> float:
        cur_lr = self.beta * self.lr
        self.lr *= self.beta
        return cur_lr

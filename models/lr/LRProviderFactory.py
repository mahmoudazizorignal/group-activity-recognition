from models.lr.providers import CosineLR, ExponentialLR
from models.lr.LREnums import LREnums
from helpers.config import Settings

class LRProviderFactory:
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def create(self, provider: LREnums):
        if provider.name == LREnums.COSINE.name:
            return CosineLR(
                max_steps=self.settings.MAX_STEPS,
                warmup_steps=self.settings.WARMUP_STEPS,
                max_lr=self.settings.MAX_LR,
                min_lr=self.settings.MIN_LR,
            )
        
        elif provider.name == LREnums.EXPONENTIAL.name:
            return ExponentialLR(
                initial_lr=self.settings.INITIAL_LR,
                beta=self.settings.BETA,
            )

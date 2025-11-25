from helpers.config import Settings
from models.baselines.providers import B1ModelProvider, B3ModelProvider
from models.baselines.BaselinesEnums import BaselinesEnums

class BaselinesProviderFactory:
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def create(self, provider: BaselinesEnums, resnet_pretrained: bool):
        if provider.name == BaselinesEnums.B1MODEL.name:
            return B1ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.B3MODEL.name:
            return B3ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained
            )

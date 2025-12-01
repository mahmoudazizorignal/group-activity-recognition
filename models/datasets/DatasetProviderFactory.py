from helpers.config import Settings
from models.datasets.providers import PersonDatasetProvider, GroupDatasetProvider
from models.datasets.DatasetEnums import DatasetEnums

class DatasetProviderFactory:
    
    def __init__(self, settings: Settings, annotations: dict):
        self.settings = settings
        self.annotations = annotations
    
    def create(self, provider: DatasetEnums, videos_split: list,):
        
        if provider.name == DatasetEnums.GROUP.name:
            return GroupDatasetProvider(
                videos_split=videos_split, 
                annotations=self.annotations, 
                settings=self.settings,
            )
        
        elif provider.name == DatasetEnums.PERSON.name:
            return PersonDatasetProvider(
                videos_split=videos_split, 
                annotations=self.annotations, 
                settings=self.settings,
            )

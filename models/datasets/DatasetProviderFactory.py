from helpers.config import Settings
from models.datasets.providers import PersonDatasetProvider, GroupDatasetProvider
from models.datasets.DatasetEnums import DatasetEnums

class DatasetProviderFactory:
    """
    Factory class for creating dataset provider instances.
    
    This factory implements the Factory design pattern to instantiate the appropriate
    dataset provider based on the specified dataset type. It encapsulates the creation
    logic for different dataset providers (group-level or person-level) and ensures
    consistent initialization with shared settings and annotations.
    
    Attributes:
        settings (Settings): Configuration object containing dataset parameters, paths,
            and model settings.
        annotations (dict): Dictionary containing the processed annotations for all videos,
            including group activities and player tracking information.
    
    Args:
        settings (Settings): Configuration object for dataset initialization.
        annotations (dict): Preprocessed annotations dictionary with structure:
            {video_no: {frame_no: {"group_activity": str, "players": list}}}
    """
    def __init__(self, settings: Settings, annotations: dict):
        self.settings = settings
        self.annotations = annotations
    
    def create(self, provider: DatasetEnums, videos_split: list,):
        """
        Create and return a dataset provider instance based on the specified type.
        
        Instantiates either a GroupDatasetProvider or PersonDatasetProvider based on the
        provided DatasetEnums value. The created provider is initialized with the factory's
        settings, annotations, and the specified video split.
        
        Args:
            provider (DatasetEnums): Enum value specifying the type of dataset provider
                to create. Should be either DatasetEnums.GROUP for group-level activity
                recognition or DatasetEnums.PERSON for person-level activity recognition.
            videos_split (list): List of video identifiers to include in the created
                dataset provider (e.g., train split, validation split, or test split).
        
        Returns:
            DatasetInterface: An instance of either GroupDatasetProvider or
                PersonDatasetProvider, depending on the provider argument. Both inherit
                from DatasetInterface and can be used as PyTorch Dataset objects.
        
        Raises:
            AttributeError: If the provider enum value does not match GROUP or PERSON,
                the method will not return a provider (implicitly returns None).
        """        
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

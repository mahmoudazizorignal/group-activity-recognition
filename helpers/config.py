import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_PATH: str = os.path.dirname(os.path.dirname(__file__))
    ANNOTATION_PATH: str
    DATASET_PATH: str

    TRAIN_VIDEOS: list
    VALIDATION_VIDEOS: list
    TEST_VIDEOS: list
    
    CNT_BEFORE_TARGET: int
    CNT_AFTER_TARGET: int

    GROUP_ACTION_CNT: int
    GROUP_ACTION_TO_ID: dict[str, int]
    PLAYER_ACTION_CNT: int
    PLAYER_ACTION_TO_ID: dict[str, int]

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
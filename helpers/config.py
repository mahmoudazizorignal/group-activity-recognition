import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_PATH: str = os.path.dirname(os.path.dirname(__file__))
    TRACKING_ANNOTATION: str
    DATASET_PATH: str
    ANNOTATION_PATH: str

    TRAIN_VIDEOS: list
    VALIDATION_VIDEOS: list
    TEST_VIDEOS: list
    
    CNT_BEFORE_TARGET: int
    CNT_AFTER_TARGET: int

    GROUP_ACTION_CNT: int
    GROUP_ACTION_TO_ID: dict[str, int]
    PLAYER_ACTION_CNT: int
    PLAYER_ACTION_TO_ID: dict[str, int]

    DEVICE: str
    MATMUL_PRECISION: str
    SEED: int
    TOTAL_BATCH: int
    MINI_BATCH: int
    GRAD_ACCUM_STEPS: int
    PIN_MEMORY: int
    NUM_WORKERS_TRAIN: int
    NUM_WORKERS_EVAL: int
    MAX_LR: float
    MIN_LR: float
    WARMUP_STEPS: float
    MAX_STEPS: int
    EVAL_STEPS: int

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
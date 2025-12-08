import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_PATH: str = os.path.dirname(os.path.dirname(__file__))
    TRACKING_ANNOTATION: str
    DATASET_PATH: str
    ANNOTATION_PATH: str
    TENSORBOARD_PATH: str

    TRAIN_VIDEOS: list
    VALIDATION_VIDEOS: list
    TEST_VIDEOS: list
    
    CNT_BEFORE_TARGET: int
    CNT_AFTER_TARGET: int
    FRAME_CNT: int
    PLAYER_CNT: int
    GROUP_ACTION_CNT: int
    GROUP_ACTION_TO_ID: dict[str, int]
    PLAYER_ACTION_CNT: int
    PLAYER_ACTION_TO_ID: dict[str, int]
    C: int
    H: int
    W: int
    NORM_MEAN: list
    NORM_STD: list
    
    DEVICE: str
    MATMUL_PRECISION: str
    SEED: int
    NUM_EPOCHS: int
    MINI_BATCH: int
    GRAD_ACCUM_STEPS: int
    EVAL_INTERVALS: int
    PIN_MEMORY: int
    NUM_WORKERS_TRAIN: int
    NUM_WORKERS_EVAL: int
    PERSISTANT_WORKERS: int
    MAX_LR: float
    MIN_LR: float
    WARMUP_STEPS: int
    MAX_STEPS: int
    INITIAL_LR: float
    BETA: float
    NO_LSTM_LAYERS: int
    NO_LSTM_HIDDEN_UNITS: int
    LSTM_DROPOUT_RATE: float
    HEAD_DROPOUT_RATE: float
    
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
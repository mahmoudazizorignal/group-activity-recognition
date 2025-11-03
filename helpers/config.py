from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    TRAIN_VIDEOS: list
    VALIDATION_VIDEOS: list

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
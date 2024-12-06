from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Service settings
    SERVICE_PORT: int = 8000
    SERVICE_HOST: str = "0.0.0.0"

    # Model settings
    MODEL_UPDATE_INTERVAL: int = 21600  # 6 hours
    INTERACTION_WEIGHT: float = 0.3
    CLEANUP_INTERVAL: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

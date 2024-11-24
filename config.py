from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database settings
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/postgres"

    # Cache settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 1800  # Cache for 30 minutes

    # Model settings
    MODEL_UPDATE_INTERVAL: int = 3600 * 6  # Update every 6 hours
    EMBEDDING_DIM: int = 32
    BATCH_SIZE: int = 100
    MAX_FEATURES: int = 100

    # Recommendation settings
    MIN_INTERACTIONS: int = 10  # Minimum interactions before using collaborative filtering

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

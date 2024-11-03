from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    REDIS_URL: str = "redis://localhost:6379"
    MODEL_UPDATE_INTERVAL: int = 3600 * 6  # Update every 6 hours
    CACHE_TTL: int = 1800  # Cache for 30 minutes
    BATCH_SIZE: int = 100
    MAX_FEATURES: int = 100


settings = Settings()

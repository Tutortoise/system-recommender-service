from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    REDIS_URL: str = "redis://localhost:6379"
    MODEL_UPDATE_INTERVAL: int = 3600  # Update model every hour
    CACHE_TTL: int = 300  # Cache recommendations for 5 minutes
    EMBEDDING_DIM: int = 32
    MIN_INTERACTIONS: int = (
        10  # Minimum interactions before using collaborative filtering
    )


settings = Settings()

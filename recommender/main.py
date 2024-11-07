import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from recommender.service import RecommenderService
from recommender.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global recommender_service
    recommender_service = RecommenderService()
    await recommender_service.initialize()
    yield
    # Shutdown
    if recommender_service.pg_pool:
        await recommender_service.pg_pool.close()
    if recommender_service.redis:
        await recommender_service.redis.close()


app = FastAPI(title="Mentor Recommendation API", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InteractionCreate(BaseModel):
    user_id: str
    mentor_id: str
    interaction_type: str
    rating: Optional[float] = None


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, top_k: int = 5):
    try:
        recommendations = await recommender_service.get_recommendations(user_id, top_k)
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        # Check database connection
        async with recommender_service.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        # Check Redis connection
        await recommender_service.redis.ping()

        # Check model status
        model_status = "ready" if recommender_service._model_ready else "not_ready"

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "redis": "connected",
            "model_status": model_status,
            "is_updating": recommender_service.is_updating,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            },
        )


if __name__ == "__main__":
    uvicorn.run("recommender.main:app", host="0.0.0.0", port=8000, reload=True)

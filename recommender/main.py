import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from recommender.service import RecommenderService


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


app = FastAPI(title="Mentor Recommendation API", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, top_k: int = 5):
    try:
        data = await recommender_service.get_recommendations(user_id, top_k)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": data,
                "code": "RECOMMENDATIONS_FOUND",
            },
        )
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "error": str(e),
            },
        )


@app.post("/model/update")
async def trigger_model_update(background_tasks: BackgroundTasks):
    """Manually trigger a model update."""
    if recommender_service.is_updating:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "code": "UPDATE_IN_PROGRESS",
                "message": "Model update already in progress",
            },
        )

    background_tasks.add_task(recommender_service.update_model)
    return JSONResponse(
        status_code=202,
        content={
            "status": "success",
            "code": "UPDATE_SCHEDULED",
            "message": "Model update scheduled",
        },
    )


@app.post("/model/reset")
async def reset_model():
    """Reset model state and clear caches"""
    try:
        # Clear caches
        recommender_service.recommender.similarity_cache.clear()

        # Reset states
        recommender_service._model_ready = False
        recommender_service.is_updating = False

        # Trigger new model update
        await recommender_service.trigger_model_update()

        return {
            "status": "success",
            "message": "Model reset and update triggered",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def get_model_status():
    """
    Get the current status of the recommendation model
    """
    return {
        "is_ready": recommender_service._model_ready,
        "is_updating": recommender_service.is_updating,
        "feature_dimensions": recommender_service.feature_processor.feature_dim,
        "total_users": len(recommender_service.recommender.user_features),
        "total_mentors": len(recommender_service.recommender.mentor_features),
        "cache_size": len(recommender_service.recommender.similarity_cache),
    }


@app.get("/health")
async def health_check():
    try:
        # Properly acquire and use connection
        async with recommender_service.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "code": "HEALTHY",
                "data": {
                    "database": "connected",
                    "model_status": (
                        "ready" if recommender_service._model_ready else "not_ready"
                    ),
                    "is_updating": recommender_service.is_updating,
                },
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "code": "UNHEALTHY",
                "message": f"Database connection failed: {str(e)}",
                "error": type(e).__name__,
            },
        )


if __name__ == "__main__":
    uvicorn.run("recommender.main:app", host="0.0.0.0", port=8000, reload=True)

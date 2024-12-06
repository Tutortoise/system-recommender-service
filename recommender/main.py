from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
async def get_recommendations(
    user_id: str,
    top_k: int = 5,
    strict: bool = False,
):
    try:
        data = await recommender_service.get_recommendations(user_id, top_k)
        if strict and data["total_found"] < top_k:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "INSUFFICIENT_RECOMMENDATIONS",
                    "message": f"Could only find {data['total_found']} recommendations of {top_k} requested",
                },
            )
        return JSONResponse(
            status_code=200, content={"status": "success", "data": data}
        )
    except HTTPException:
        raise


@app.get("/interaction/{user_id}/{tutories_id}")
async def track_interaction(
    user_id: str,
    tutories_id: str,
):
    """Track user clicks on tutories"""
    try:
        async with recommender_service.pg_pool.acquire() as conn:
            tutory = await conn.fetchrow(
                """
                SELECT
                    ty.id as tutory_id,
                    ty.tutor_id,
                    ty.category_id,
                    c.name as category_name
                FROM tutories ty
                JOIN categories c ON ty.category_id = c.id
                WHERE ty.id = $1
                """,
                tutories_id,
            )

            if not tutory:
                raise HTTPException(status_code=404, detail="Tutory not found")

            success = recommender_service.recommender.process_interaction(
                user_id=user_id,
                tutory_id=tutories_id,
                tutor_id=str(tutory["tutor_id"]),
                category_id=str(tutory["category_id"]),
                category_name=tutory["category_name"],
            )

            return {
                "status": "success" if success else "failed",
                "message": "Click interaction recorded",
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """Get the current status of the recommendation model"""
    tracker = recommender_service.recommender.interaction_tracker

    return {
        "is_ready": recommender_service._model_ready,
        "is_updating": recommender_service.is_updating,
        "feature_dimensions": recommender_service.feature_processor.feature_dim,
        "total_users": len(recommender_service.recommender.user_features),
        "total_mentors": len(recommender_service.recommender.mentor_features),
        "cache_size": len(recommender_service.recommender.similarity_cache),
        "interaction_stats": {
            "total_users_with_interactions": len(tracker.interactions),
            "total_interactions": sum(
                len(ints) for ints in tracker.interactions.values()
            ),
        },
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

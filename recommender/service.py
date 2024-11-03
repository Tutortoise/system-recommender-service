import asyncio
from typing import List, Optional
import asyncpg
import redis.asyncio as redis
from datetime import datetime
from collections import defaultdict
import json
import logging

from .config import settings
from .feature_processor import FeatureProcessor
from .tf_recommender import TensorFlowRecommender


class RecommenderService:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.recommender = TensorFlowRecommender()
        self.redis = None
        self.pg_pool = None
        self.is_updating = False
        self._model_ready = False
        self._lock = asyncio.Lock()
        self.interaction_counts = defaultdict(int)

    async def initialize(self):
        self.redis = await redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.pg_pool = await asyncpg.create_pool(
            settings.POSTGRES_URL, min_size=5, max_size=20
        )

        # Initial model training
        async with self._lock:
            await self.update_model()
            self._model_ready = True

        # Start background update task
        asyncio.create_task(self.periodic_model_update())

    async def get_recommendations(self, user_id: str, top_k: int = 5) -> List[str]:
        if not self._model_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

        try:
            # Check if user exists
            async with self.pg_pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT * FROM user_features WHERE user_id = $1", user_id
                )
                if not user:
                    raise ValueError(f"User {user_id} not found")

            # Try cache first
            cache_key = f"recommendations:{user_id}"
            cached = await self.redis.get(cache_key)
            if cached:
                return cached.split(",")

            # Process user features
            user_vector = self.feature_processor.process_user_features(dict(user))
            self.recommender.update_single_user(user_id, user_vector)

            # Get recommendations
            recommendations = self.recommender.get_recommendations(user_id, top_k)

            # Cache results
            if recommendations:
                await self.redis.set(cache_key, ",".join(recommendations))
                await self.redis.expire(cache_key, settings.CACHE_TTL)

            return recommendations

        except ValueError as e:
            logging.error(f"User not found: {user_id}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    async def update_model(self):
        if self.is_updating:
            return

        self.is_updating = True
        try:
            async with self.pg_pool.acquire() as conn1, self.pg_pool.acquire() as conn2:
                users = await conn1.fetch("SELECT * FROM user_features")
                mentors = await conn2.fetch("SELECT * FROM mentor_features")

            # Process features in batches
            batch_size = settings.BATCH_SIZE
            all_texts = []

            # Process in chunks to avoid memory issues
            for i in range(0, len(users), batch_size):
                batch = users[i : i + batch_size]
                all_texts.extend([" ".join(user["interests"]) for user in batch])

            for i in range(0, len(mentors), batch_size):
                batch = mentors[i : i + batch_size]
                all_texts.extend([" ".join(mentor["expertise"]) for mentor in batch])

            # Update feature processor
            self.feature_processor.fit(all_texts)

            # Process features in batches
            user_features = {}
            mentor_features = {}

            for i in range(0, len(users), batch_size):
                batch = users[i : i + batch_size]
                for user in batch:
                    user_features[str(user["user_id"])] = (
                        self.feature_processor.process_user_features(dict(user))
                    )

            for i in range(0, len(mentors), batch_size):
                batch = mentors[i : i + batch_size]
                for mentor in batch:
                    mentor_features[str(mentor["mentor_id"])] = (
                        self.feature_processor.process_mentor_features(dict(mentor))
                    )

            # Update fast recommender
            self.recommender.build_index(mentor_features)
            self.recommender.update_user_features(user_features)

        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            self._model_ready = False
            raise
        finally:
            self.is_updating = False

    async def record_interaction(
        self,
        user_id: str,
        mentor_id: str,
        interaction_type: str,
        rating: Optional[float] = None,
    ):
        interaction_data = {
            "user_id": user_id,
            "mentor_id": mentor_id,
            "interaction_type": interaction_type,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in Redis for quick access
        interaction_key = f"interaction:{user_id}:{mentor_id}"
        await self.redis.lpush(interaction_key, json.dumps(interaction_data))

        # Update interaction counts
        self.interaction_counts[user_id] += 1

        # Store in PostgreSQL for persistence
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interactions
                (user_id, mentor_id, interaction_type, rating, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """,
                user_id,
                mentor_id,
                interaction_type,
                rating,
                datetime.utcnow(),
            )

    async def trigger_model_update(self):
        if not self.is_updating:
            self.is_updating = True
            try:
                await self.update_model()
            finally:
                self.is_updating = False

    async def periodic_model_update(self):
        while True:
            try:
                async with self._lock:
                    await self.update_model()
            except Exception as e:
                logging.error(f"Error in periodic model update: {str(e)}")
            finally:
                await asyncio.sleep(settings.MODEL_UPDATE_INTERVAL)

import asyncio
from typing import List, Dict
import asyncpg
import logging
from uuid import UUID
from fastapi import HTTPException

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import settings
from .feature_processor import FeatureProcessor
from .recommender import Recommender


class RecommenderService:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.recommender = Recommender()
        self.pg_pool = None
        self.is_updating = False
        self._model_ready = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        self.pg_pool = await asyncpg.create_pool(
            settings.POSTGRES_URL,
            min_size=5,
            max_size=20,
            command_timeout=60,
            init=lambda conn: conn.execute(
                """
                SET work_mem = '32MB';
                SET random_page_cost = 1.1;
                SET effective_cache_size = '1GB';
            """
            ),
        )

        # Initial model training
        async with self._lock:
            await self.update_model()
            self._model_ready = True

        # Start background update task
        asyncio.create_task(self.periodic_model_update())

    async def get_recommendations(self, user_id: str, top_k: int = 5) -> Dict:
        try:
            UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_UUID",
                    "message": "Invalid user ID format. Must be a valid UUID.",
                    "user_id": user_id,
                },
            )

        if not self._model_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Recommendation service is not ready",
                },
            )

        try:
            async with self.pg_pool.acquire() as conn:
                # Get user data with more details
                user = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        email,
                        interest,
                        learning_style,
                        city
                    FROM users
                    WHERE id = $1
                """,
                    user_id,
                )

                if not user:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "code": "USER_NOT_FOUND",
                            "message": f"User with ID {user_id} not found",
                            "user_id": user_id,
                        },
                    )

                user_vector = self.feature_processor.process_user_features(
                    {
                        "interest": user["interest"],
                        "learning_style": user["learning_style"],
                        "city": user["city"],
                    }
                )

                self.recommender.update_single_user(user_id, user_vector)
                tutor_ids = self.recommender.get_recommendations(user_id, top_k)
                tutor_ids_str = [str(tid) for tid in tutor_ids]

                recommended_tutors = await conn.fetch(
                    """
                    WITH RECURSIVE
                    user_interests AS (
                        SELECT DISTINCT LOWER(unnest(interest)) as interest
                        FROM users
                        WHERE id = $1
                    ),
                    filtered_tutors AS (
                        -- Pre-filter tutors and compute matches
                        SELECT DISTINCT ON (t.id, ts.subject)  -- Get unique tutor-subject combinations
                            t.id::text,
                            t.email,
                            ts.id as service_id,
                            ts.subject,
                            ts.specialization,
                            ts.teaching_style,
                            ts.hourly_rate::float,
                            t.latitude,
                            t.longitude,
                            COALESCE(AVG(sr.session_rating) OVER (PARTITION BY t.id, ts.subject), 0.0)::float as average_rating,
                            COUNT(sr.session_rating) OVER (PARTITION BY t.id, ts.subject)::int as num_ratings,
                            CASE
                                WHEN EXISTS (
                                    SELECT 1 FROM user_interests ui
                                    WHERE LOWER(ts.subject) = ui.interest
                                ) THEN 3  -- Exact match
                                WHEN EXISTS (
                                    SELECT 1 FROM user_interests ui
                                    WHERE LOWER(ts.subject) LIKE '%' || ui.interest || '%'
                                    OR ui.interest LIKE '%' || LOWER(ts.subject) || '%'
                                ) THEN 2  -- Partial match
                                ELSE 1    -- No direct match
                            END as match_score
                        FROM tutor t
                        INNER JOIN tutor_service ts ON t.id = ts.tutor_id
                        LEFT JOIN session_rating sr ON ts.id = sr.service_id
                        WHERE t.id::text = ANY($2)
                    ),
                    ranked_results AS (
                        -- Apply ranking with window functions
                        SELECT
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY subject
                                ORDER BY
                                    match_score DESC,
                                    average_rating DESC,
                                    num_ratings DESC
                            ) as subject_rank,
                            CASE
                                WHEN average_rating >= 4.5 AND num_ratings >= 10 THEN 1
                                WHEN average_rating >= 4.0 AND num_ratings >= 5 THEN 2
                                WHEN average_rating >= 3.5 THEN 3
                                ELSE 4
                            END as rating_tier
                        FROM filtered_tutors
                    )
                    SELECT
                        id,
                        email,
                        service_id,
                        subject,
                        specialization,
                        teaching_style,
                        hourly_rate,
                        average_rating,
                        num_ratings,
                        match_score,
                        subject_rank,
                        rating_tier
                    FROM ranked_results
                    ORDER BY
                        match_score DESC,               -- Best matches first
                        rating_tier ASC,                -- Higher rated tutors
                        average_rating DESC,            -- Specific rating
                        num_ratings DESC,               -- Number of ratings
                        subject_rank ASC                -- Best within subject
                    FETCH FIRST $3 ROWS WITH TIES      -- Get top-k with ties
                    """,
                    user_id,
                    tutor_ids_str,
                    top_k,
                )

                # Process recommendations
                recommendations = []
                seen_combinations = set()  # Track unique tutor-subject combinations

                for tutor in recommended_tutors:
                    match_reasons = self._get_match_reasons(
                        user["interest"],
                        user["learning_style"],
                        tutor["subject"],
                        tutor["teaching_style"],
                        tutor["average_rating"],
                    )

                    if not match_reasons:
                        continue

                    # Add rating-based reason if applicable
                    if tutor["average_rating"] >= 4.5 and tutor["num_ratings"] >= 10:
                        match_reasons.append(
                            f"Highly rated tutor ({tutor['average_rating']}/5 from {tutor['num_ratings']} reviews)"
                        )
                    elif tutor["average_rating"] >= 4.0 and tutor["num_ratings"] >= 5:
                        match_reasons.append(
                            f"Well-rated tutor ({tutor['average_rating']}/5)"
                        )

                    recommendation = {
                        "id": str(tutor["id"]),
                        "service_id": str(tutor["service_id"]),
                        "email": tutor["email"],
                        "subject": tutor["subject"],
                        "specialization": tutor["specialization"],
                        "teaching_style": tutor["teaching_style"],
                        "hourly_rate": float(tutor["hourly_rate"]),
                        "average_rating": float(tutor["average_rating"]),
                        "num_ratings": int(tutor["num_ratings"]),
                        "match_reasons": match_reasons,
                        "match_score": tutor["match_score"],
                        "subject_rank": tutor["subject_rank"],
                    }

                    recommendations.append(recommendation)

                # Sort final recommendations
                recommendations.sort(
                    key=lambda x: (
                        -x["match_score"],  # Higher match score first
                        -x["average_rating"],  # Higher rating next
                        -x["num_ratings"],  # More ratings next
                        x["subject_rank"],  # Better ranked within subject
                    )
                )

                response = {
                    "user": {
                        "id": str(user["id"]),
                        "email": user["email"],
                        "interests": user["interest"],
                        "learning_style": user["learning_style"],
                        "city": user["city"],
                    },
                    "recommendations": recommendations[:top_k],
                }

                return response

        except HTTPException:
            raise

        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                },
            )

    def _get_match_reasons(
        self,
        user_interests: List[str],
        user_style: str,
        tutor_subject: str,
        tutor_style: str,
        rating: float,
    ) -> List[str]:
        reasons = []
        similarities = []

        for interest in user_interests:
            sim = self.feature_processor.subject_embedding.get_subject_similarity(
                interest, tutor_subject
            )
            similarities.append((interest, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        if similarities:
            best_match, best_sim = similarities[0]
            if best_sim > 0.85:
                reasons.append(
                    f"Perfect match: Teaches {tutor_subject} (exactly matches your interest in {best_match})"
                )
            elif best_sim > 0.7:
                reasons.append(
                    f"Strong match: {tutor_subject} is closely related to your interest in {best_match}"
                )
            elif best_sim > 0.5:
                reasons.append(f"Related to your interest in {best_match}")
            else:
                return []  # Don't include unrelated subjects

        return reasons

    async def update_model(self):
        try:
            self.recommender.reset_caches()

            async with self.pg_pool.acquire() as conn:
                # Fetch users
                users = await conn.fetch(
                    """
                    SELECT
                        id::text as user_id,
                        interest,
                        learning_style,
                        city
                    FROM users
                """
                )

                tutors = await conn.fetch(
                    """
                    SELECT
                        t.id::text as tutor_id,
                        t.latitude,
                        t.longitude,
                        ts.teaching_style,
                        ts.subject,
                        ts.specialization,
                        ts.hourly_rate::float as hourly_rate,
                        ts.year_of_experience::int as year_of_experience,
                        COALESCE(AVG(sr.session_rating)::float, 0.0) as average_rating,
                        COUNT(sr.session_rating)::int as num_ratings
                    FROM tutor t
                    JOIN tutor_service ts ON t.id = ts.tutor_id
                    LEFT JOIN session_rating sr ON ts.id = sr.service_id
                    GROUP BY t.id, t.latitude, t.longitude, ts.id, ts.teaching_style,
                             ts.subject, ts.specialization, ts.hourly_rate,
                             ts.year_of_experience
                    ORDER BY average_rating DESC, num_ratings DESC
                """
                )

                # Process features
                all_texts = []
                for user in users:
                    if user["interest"]:
                        all_texts.append(" ".join(user["interest"]))

                for tutor in tutors:
                    if tutor["specialization"]:
                        all_texts.append(" ".join(tutor["specialization"]))
                    if tutor["subject"]:
                        all_texts.append(tutor["subject"])

                self.feature_processor.fit(all_texts)

                # Process features with proper type handling
                user_features = {}
                tutor_features = {}

                for user in users:
                    user_features[user["user_id"]] = (
                        self.feature_processor.process_user_features(
                            {
                                "interest": user["interest"],
                                "learning_style": user["learning_style"],
                                "city": user["city"],
                            }
                        )
                    )

                for tutor in tutors:
                    rating_data = {
                        "average_rating": float(tutor["average_rating"]),
                        "num_ratings": int(tutor["num_ratings"]),
                    }

                    location = (
                        (float(tutor["latitude"]), float(tutor["longitude"]))
                        if tutor["latitude"] is not None
                        and tutor["longitude"] is not None
                        else None
                    )

                    tutor_features[tutor["tutor_id"]] = (
                        self.feature_processor.process_tutor_features(
                            {"location": location},
                            {
                                "teaching_style": tutor["teaching_style"],
                                "subject": tutor["subject"],
                                "specialization": tutor["specialization"],
                                "hourly_rate": float(tutor["hourly_rate"]),
                                "year_of_experience": int(tutor["year_of_experience"]),
                            },
                            rating_data,
                        )
                    )

                # Update recommender
                self.recommender.build_index(tutor_features)
                self.recommender.update_user_features(user_features)

                self._model_ready = True

        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            self._model_ready = False
            raise
        finally:
            self.is_updating = False

    async def trigger_model_update(self):
        """Trigger a manual model update"""
        try:
            async with self._lock:  # Ensure thread safety
                if self.is_updating:
                    raise ValueError("Model update already in progress")

                self.is_updating = True
                logging.info("Starting manual model update...")

                self.recommender.reset_caches()
                await self.update_model()

                return {
                    "status": "success",
                    "is_ready": self._model_ready,
                }

        except Exception as e:
            logging.error(f"Model update failed: {str(e)}")
            raise
        finally:
            self.is_updating = False

    async def periodic_model_update(self):
        while True:
            try:
                async with self._lock:
                    self.recommender.reset_caches()
                    await self.update_model()
            except Exception as e:
                logging.error(f"Error in periodic model update: {str(e)}")
            finally:
                await asyncio.sleep(settings.MODEL_UPDATE_INTERVAL)

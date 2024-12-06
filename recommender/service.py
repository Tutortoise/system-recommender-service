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

        self.cleanup_interval = settings.CLEANUP_INTERVAL
        asyncio.create_task(self.periodic_cleanup())

    async def initialize(self):
        self.pg_pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
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
            # Try to rebuild the model if it's not ready
            await self.update_model()
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
                # Get learner data
                learner = await conn.fetchrow(
                    """
                    WITH learner_interests AS (
                        SELECT
                            i.learner_id,
                            array_agg(c.name) as interests
                        FROM interests i
                        JOIN categories c ON i.category_id = c.id
                        GROUP BY i.learner_id
                    )
                    SELECT
                        l.id,
                        l.email,
                        l.name,
                        l.learning_style,
                        l.gender,
                        l.city,
                        l.district,
                        li.interests
                    FROM learners l
                    LEFT JOIN learner_interests li ON l.id = li.learner_id
                    WHERE l.id = $1
                    """,
                    user_id,
                )

                if not learner:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "code": "LEARNER_NOT_FOUND",
                            "message": f"Learner with ID {user_id} not found",
                            "user_id": user_id,
                        },
                    )

                # Process learner features
                user_vector = self.feature_processor.process_user_features(
                    {
                        "learning_style": learner["learning_style"],
                        "city": learner["city"],
                        "district": learner["district"],
                        "interests": learner["interests"] or [],
                    }
                )

                buffer_factor = 3  # Get 3x more recommendations initially
                initial_top_k = top_k * buffer_factor

                self.recommender.update_single_user(user_id, user_vector)
                tutor_ids = self.recommender.get_recommendations(user_id, initial_top_k)
                tutor_ids_str = [str(tid) for tid in tutor_ids]

                # Get recommended tutors with their tutories
                recommended_tutors = await conn.fetch(
                    """
                    WITH tutor_stats AS (
                        SELECT
                            t.id as tutor_id,
                            t.name,
                            t.email,
                            t.gender,
                            t.city,
                            t.district,
                            ty.id as tutories_id,
                            c.name as category_name,
                            ty.name as tutory_name,
                            ty.about_you,
                            ty.teaching_methodology,
                            ty.hourly_rate,
                            ty.type_lesson,
                            COUNT(o.id) as total_orders,
                            COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders,
                            COALESCE(AVG(r.rating), 0) as avg_rating,
                            COUNT(r.id) as review_count
                        FROM tutors t
                        INNER JOIN tutories ty ON t.id = ty.tutor_id
                        INNER JOIN categories c ON ty.category_id = c.id
                        LEFT JOIN orders o ON ty.id = o.tutories_id
                        LEFT JOIN reviews r ON o.id = r.order_id
                        WHERE t.id::text = ANY($1)
                        AND ty.is_enabled = true
                        AND (
                            ty.type_lesson IN ('online', 'both')
                            OR
                            (ty.type_lesson = 'offline' AND t.city = $2)
                        )
                        GROUP BY
                            t.id, t.name, t.email, t.gender, t.city, t.district,
                            ty.id, c.name, ty.name, ty.about_you, ty.teaching_methodology,
                            ty.hourly_rate, ty.type_lesson
                    )
                    SELECT *
                    FROM tutor_stats
                    ORDER BY completed_orders DESC, total_orders DESC
                    LIMIT $3
                    """,
                    tutor_ids_str,
                    learner["city"],
                    top_k,
                )

                seen_tutors = set()
                recommendations = []

                for tutor in recommended_tutors:
                    tutor_id = str(tutor["tutor_id"])

                    if tutor_id in seen_tutors:
                        continue
                    seen_tutors.add(tutor_id)

                    location_match = self._calculate_location_match(
                        {"city": learner["city"], "district": learner["district"]},
                        {"city": tutor["city"], "district": tutor["district"]},
                    )

                    lesson_type = tutor["type_lesson"].lower()
                    availability_info = {
                        "can_teach_online": lesson_type in ["online", "both"],
                        "can_teach_offline": lesson_type in ["offline", "both"],
                        "same_city": tutor["city"].lower() == learner["city"].lower(),
                    }

                    match_reasons = self.feature_processor._get_match_reasons(
                        learner,
                        tutor,
                        {
                            "hourly_rate": tutor["hourly_rate"],
                            "type_lesson": tutor["type_lesson"],
                            "category_name": tutor["category_name"],
                        },
                    )

                    # Add availability-specific reasons
                    if availability_info["can_teach_online"]:
                        match_reasons.append("Available for online lessons")
                    if (
                        availability_info["can_teach_offline"]
                        and availability_info["same_city"]
                    ):
                        match_reasons.append(
                            "Available for in-person lessons in your city"
                        )

                    recommendation = {
                        "tutor_id": str(tutor["tutor_id"]),
                        "tutories_id": str(tutor["tutories_id"]),
                        "name": tutor["name"],
                        "email": tutor["email"],
                        "city": tutor["city"],
                        "district": tutor["district"],
                        "category": tutor["category_name"],
                        "tutory_name": tutor["tutory_name"],
                        "about": tutor["about_you"],
                        "methodology": tutor["teaching_methodology"],
                        "hourly_rate": float(tutor["hourly_rate"]),
                        "type_lesson": tutor["type_lesson"],
                        "completed_orders": int(tutor["completed_orders"]),
                        "total_orders": int(tutor["total_orders"]),
                        "match_reasons": match_reasons,
                        "location_match": location_match,
                        "availability": availability_info,
                    }

                    recommendations.append(recommendation)

                    if len(recommendations) >= top_k:
                        break

                if len(recommendations) < top_k:
                    additional_tutors = await self._get_additional_recommendations(
                        conn,
                        user_id,
                        learner,
                        top_k - len(recommendations),
                        seen_tutors,
                    )

                    for tutor in additional_tutors:
                        if len(recommendations) >= top_k:
                            break

                        recommendations.append(recommendation)

                response = {
                    "learner": {
                        "id": str(learner["id"]),
                        "name": learner["name"],
                        "email": learner["email"],
                        "learning_style": learner["learning_style"],
                        "city": learner["city"],
                        "district": learner["district"],
                        "interests": learner["interests"] or [],
                    },
                    "recommendations": recommendations,
                    "total_found": len(recommendations),
                    "requested": top_k,
                }

                return response

        except HTTPException:
            raise
        except ValueError as e:
            if "Recommender is not ready" in str(e):
                await self.update_model()
                # Retry the recommendation
                tutor_ids = self.recommender.get_recommendations(user_id, top_k)
                tutor_ids_str = [str(tid) for tid in tutor_ids]
            else:
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

    async def _get_additional_recommendations(
        self, conn, user_id: str, learner: Dict, needed: int, exclude_tutors: set
    ) -> List[Dict]:
        """Get additional recommendations with relaxed constraints"""
        try:
            # Get recommendations with relaxed constraints
            additional_tutors = await conn.fetch(
                """
                SELECT
                    t.id as tutor_id,
                    -- ... (same fields as before)
                FROM tutors t
                INNER JOIN tutories ty ON t.id = ty.tutor_id
                INNER JOIN categories c ON ty.category_id = c.id
                LEFT JOIN orders o ON ty.id = o.tutories_id
                WHERE ty.is_enabled = true
                AND t.id::text NOT IN (SELECT unnest($1::text[]))
                AND (
                    ty.type_lesson = 'online'
                    OR t.city = $2
                )
                GROUP BY t.id, ty.id, c.name
                ORDER BY
                    CASE WHEN t.city = $2 THEN 0 ELSE 1 END,
                    completed_orders DESC,
                    total_orders DESC
                LIMIT $3
                """,
                list(exclude_tutors),
                learner["city"],
                needed,
            )

            return additional_tutors

        except Exception as e:
            logging.error(f"Error getting additional recommendations: {str(e)}")
            return []

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
                # First, collect all texts for fitting
                categories = await conn.fetch(
                    "SELECT name FROM categories"
                )  # Changed from subjects
                tutories = await conn.fetch(
                    """
                    SELECT teaching_methodology, about_you
                    FROM tutories
                """
                )

                # Prepare texts for fitting
                all_texts = []

                # Add category names
                for category in categories:
                    all_texts.append(category["name"])

                # Add teaching methodologies and about texts
                for tutory in tutories:
                    if tutory["teaching_methodology"]:
                        all_texts.append(tutory["teaching_methodology"])
                    if tutory["about_you"]:
                        all_texts.append(tutory["about_you"])

                # Fit the feature processor
                self.feature_processor.fit(all_texts)

                # Now fetch and process learner data with their interests
                learners = await conn.fetch(
                    """
                    WITH learner_interests AS (
                        SELECT
                            i.learner_id,
                            array_agg(c.name) as interests
                        FROM interests i
                        JOIN categories c ON i.category_id = c.id
                        GROUP BY i.learner_id
                    )
                    SELECT
                        l.id::text as learner_id,
                        l.learning_style,
                        l.city,
                        l.district,
                        li.interests
                    FROM learners l
                    LEFT JOIN learner_interests li ON l.id = li.learner_id
                """
                )

                # Fetch tutor data with statistics
                tutors = await conn.fetch(
                    """
                    WITH tutor_stats AS (
                        SELECT
                            t.id as tutor_id,
                            ty.id as tutories_id,
                            c.name as category_name,
                            ty.name as tutory_name,
                            ty.teaching_methodology,
                            ty.hourly_rate,
                            ty.type_lesson,
                            t.city,
                            t.district,
                            COUNT(o.id) as total_orders,
                            COUNT(o.id) FILTER (WHERE o.status = 'completed') as completed_orders,
                            AVG(r.rating) as avg_rating,
                            COUNT(r.id) as review_count
                        FROM tutors t
                        JOIN tutories ty ON t.id = ty.tutor_id
                        JOIN categories c ON ty.category_id = c.id
                        LEFT JOIN orders o ON ty.id = o.tutories_id
                        LEFT JOIN reviews r ON o.id = r.order_id
                        WHERE ty.is_enabled = true
                        GROUP BY t.id, ty.id, c.name
                    )
                    SELECT
                        *,
                        COALESCE(completed_orders::float / NULLIF(total_orders, 0), 0) as completion_rate
                    FROM tutor_stats
                """
                )

                # Process features
                learner_features = {}
                for learner in learners:
                    learner_features[learner["learner_id"]] = (
                        self.feature_processor.process_user_features(
                            {
                                "learning_style": learner["learning_style"],
                                "city": learner["city"],
                                "district": learner["district"],
                                "interests": learner["interests"],
                            }
                        )
                    )

                tutor_features = {}
                for tutor in tutors:
                    tutor_features[tutor["tutor_id"]] = (
                        self.feature_processor.process_tutor_features(
                            {"city": tutor["city"], "district": tutor["district"]},
                            {
                                "subject": tutor[
                                    "category_name"
                                ],  # Changed from subject_name
                                "name": tutor["tutory_name"],  # Added tutory name
                                "teaching_methodology": tutor["teaching_methodology"],
                                "hourly_rate": float(tutor["hourly_rate"]),
                                "type_lesson": tutor["type_lesson"],
                            },
                            {
                                "avg_rating": float(tutor["avg_rating"] or 0),
                                "completion_rate": float(tutor["completion_rate"]),
                                "review_count": int(tutor["review_count"]),
                            },
                        )
                    )

                # Update recommender
                self.recommender.build_index(tutor_features)
                self.recommender.update_user_features(learner_features)

                self._model_ready = True

        except Exception as e:
            logging.error(f"Error updating model: {str(e)}", exc_info=True)
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

    def _calculate_location_match(self, learner: Dict, tutor: Dict) -> float:
        """Calculate location match score"""
        if learner["city"] == tutor["city"]:
            if learner["district"] == tutor["district"]:
                return 1.0  # Perfect match - same district
            return 0.7  # Same city, different district
        return 0.0  # Different cities

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

    async def periodic_cleanup(self):
        """Periodically clean up old interactions"""
        while True:
            try:
                self.recommender.interaction_tracker.clear_old_interactions()
            except Exception as e:
                logging.error(f"Error in interaction cleanup: {str(e)}")
            finally:
                await asyncio.sleep(self.cleanup_interval)

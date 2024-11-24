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
                learner = await conn.fetchrow("""
                    SELECT
                        id,
                        email,
                        name,
                        learning_style,
                        gender,
                        city,
                        district
                    FROM learners
                    WHERE id = $1
                """, user_id)

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
                user_vector = self.feature_processor.process_user_features({
                    "learning_style": learner["learning_style"],
                    "city": learner["city"],
                    "district": learner["district"]
                })

                self.recommender.update_single_user(user_id, user_vector)
                tutor_ids = self.recommender.get_recommendations(user_id, top_k)
                tutor_ids_str = [str(tid) for tid in tutor_ids]

                # Get recommended tutors with their tutories
                recommended_tutors = await conn.fetch("""
                    WITH tutor_stats AS (
                        SELECT
                            t.id as tutor_id,
                            t.name,
                            t.email,
                            t.gender,
                            t.city,
                            t.district,
                            ty.id as tutory_id,
                            s.name as subject_name,
                            ty.about_you,
                            ty.teaching_methodology,
                            ty.hourly_rate,
                            ty.type_lesson,
                            ty.availability,
                            COUNT(o.id) as total_orders,
                            COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders
                        FROM tutors t
                        INNER JOIN tutories ty ON t.id = ty.tutor_id
                        INNER JOIN subjects s ON ty.subject_id = s.id
                        LEFT JOIN orders o ON ty.id = o.tutory_id
                        WHERE t.id::text = ANY($1)
                        GROUP BY
                            t.id, t.name, t.email, t.gender, t.city, t.district,
                            ty.id, s.name, ty.about_you, ty.teaching_methodology,
                            ty.hourly_rate, ty.type_lesson, ty.availability
                    )
                    SELECT
                        tutor_id,
                        name,
                        email,
                        gender,
                        city,
                        district,
                        tutory_id,
                        subject_name,
                        about_you,
                        teaching_methodology,
                        hourly_rate,
                        type_lesson,
                        availability,
                        total_orders,
                        completed_orders
                    FROM tutor_stats
                    ORDER BY completed_orders DESC, total_orders DESC
                """, tutor_ids_str)

                recommendations = []
                for tutor in recommended_tutors:
                    location_match = self._calculate_location_match(
                        {
                            "city": learner["city"],
                            "district": learner["district"]
                        },
                        {
                            "city": tutor["city"],
                            "district": tutor["district"]
                        }
                    )

                    match_reasons = self.feature_processor._get_match_reasons(
                        learner,
                        tutor,
                        tutor  # tutory data is part of tutor data in our query
                    )

                    recommendation = {
                        "tutor_id": str(tutor["tutor_id"]),
                        "tutory_id": str(tutor["tutory_id"]),
                        "name": tutor["name"],
                        "email": tutor["email"],
                        "city": tutor["city"],
                        "district": tutor["district"],
                        "subject": tutor["subject_name"],
                        "about": tutor["about_you"],
                        "methodology": tutor["teaching_methodology"],
                        "hourly_rate": float(tutor["hourly_rate"]),
                        "type_lesson": tutor["type_lesson"],
                        "completed_orders": int(tutor["completed_orders"]),
                        "total_orders": int(tutor["total_orders"]),
                        "availability": tutor["availability"],
                        "match_reasons": match_reasons,
                        "location_match": location_match
                    }

                    recommendations.append(recommendation)

                response = {
                    "learner": {
                        "id": str(learner["id"]),
                        "name": learner["name"],
                        "email": learner["email"],
                        "learning_style": learner["learning_style"],
                    },
                    "recommendations": recommendations,
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
                # Fetch all text data first for fitting
                subjects = await conn.fetch("SELECT name FROM subjects")
                tutories = await conn.fetch(
                    """
                    SELECT teaching_methodology, about_you
                    FROM tutories
                """
                )

                # Prepare texts for fitting
                all_texts = []

                # Add subject names
                for subject in subjects:
                    all_texts.append(subject["name"])

                # Add teaching methodologies and about texts
                for tutory in tutories:
                    if tutory["teaching_methodology"]:
                        all_texts.append(tutory["teaching_methodology"])
                    if tutory["about_you"]:
                        all_texts.append(tutory["about_you"])

                # Fit the feature processor
                self.feature_processor.fit(all_texts)

                # Now fetch and process learner and tutor data
                learners = await conn.fetch("""
                    SELECT
                        id::text as learner_id,
                        learning_style,
                        city,
                        district
                    FROM learners
                """)

                tutors = await conn.fetch("""
                    SELECT
                        t.id::text as tutor_id,
                        t.city,
                        t.district,
                        ty.teaching_methodology,
                        s.name as subject_name,
                        ty.hourly_rate,
                        ty.type_lesson,
                        COUNT(o.id) FILTER (WHERE o.status = 'completed') as completed_orders
                    FROM tutors t
                    JOIN tutories ty ON t.id = ty.tutor_id
                    JOIN subjects s ON ty.subject_id = s.id
                    LEFT JOIN orders o ON ty.id = o.tutory_id
                    GROUP BY
                        t.id, t.city, t.district,
                        ty.teaching_methodology, s.name,
                        ty.hourly_rate, ty.type_lesson
                """)

                # Process features
                learner_features = {}
                tutor_features = {}

                for learner in learners:
                    learner_features[learner["learner_id"]] = self.feature_processor.process_user_features({
                        "learning_style": learner["learning_style"],
                        "city": learner["city"],
                        "district": learner["district"]
                    })

                for tutor in tutors:
                    tutor_features[tutor["tutor_id"]] = self.feature_processor.process_tutor_features(
                        {
                            "city": tutor["city"],
                            "district": tutor["district"]
                        },
                        {
                            "teaching_methodology": tutor["teaching_methodology"],
                            "subject": tutor["subject_name"],
                            "hourly_rate": float(tutor["hourly_rate"]),
                            "type_lesson": tutor["type_lesson"]
                        }
                    )

                # Update recommender
                self.recommender.build_index(tutor_features)
                self.recommender.update_user_features(learner_features)

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

    def _calculate_location_match(self, learner: Dict, tutor: Dict) -> float:
        """Calculate location match score"""
        if learner["city"] == tutor["city"]:
            if learner["district"] == tutor["district"]:
                return 1.0  # Perfect match - same district
            return 0.7     # Same city, different district
        return 0.0        # Different cities

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

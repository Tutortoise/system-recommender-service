import asyncio
from typing import List, Optional, Dict
import asyncpg
import redis.asyncio as redis
from datetime import datetime
from collections import defaultdict
import json
import logging

from .config import settings
from .feature_processor import FeatureProcessor
from .recommender import Recommender


class RecommenderService:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.recommender = Recommender()
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

    async def get_recommendations(self, user_id: str, top_k: int = 5) -> Dict:
        if not self._model_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

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
                    raise ValueError(f"User {user_id} not found")

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
                    WITH user_interests AS (
                        SELECT unnest(interest) as interest
                        FROM users
                        WHERE id = $1
                    ),
                    ranked_tutors AS (
                        SELECT
                            t.id::text,
                            t.email,
                            ts.subject,
                            ts.specialization,
                            ts.teaching_style,
                            ts.hourly_rate::float,
                            t.latitude,
                            t.longitude,
                            COALESCE(AVG(sr.session_rating)::float, 0.0) as average_rating,
                            COUNT(sr.session_rating)::int as num_ratings,
                            EXISTS (
                                SELECT 1 FROM user_interests
                                WHERE LOWER(interest) = LOWER(ts.subject)
                            ) as exact_match,
                            EXISTS (
                                SELECT 1 FROM user_interests
                                WHERE LOWER(ts.subject) LIKE LOWER('%' || interest || '%')
                                OR LOWER(interest) LIKE LOWER('%' || ts.subject || '%')
                            ) as partial_match,
                            array_agg(DISTINCT sr.session_rating) as ratings
                        FROM tutor t
                        JOIN tutor_service ts ON t.id = ts.tutor_id
                        LEFT JOIN session_rating sr ON ts.id = sr.service_id
                        WHERE t.id::text = ANY($2)
                        GROUP BY
                            t.id,
                            t.email,
                            ts.subject,
                            ts.specialization,
                            ts.teaching_style,
                            ts.hourly_rate,
                            t.latitude,
                            t.longitude
                    )
                    SELECT *,
                        CASE
                            WHEN exact_match THEN 1
                            WHEN partial_match THEN 2
                            ELSE 3
                        END as match_rank
                    FROM ranked_tutors
                    ORDER BY
                        match_rank ASC,                    -- Primary sort: exact subject matches first
                        array_length(specialization, 1) DESC,  -- Secondary sort: more specializations
                        CASE
                            WHEN average_rating >= 4.0 AND num_ratings >= 3 THEN 1
                            WHEN average_rating >= 3.5 THEN 2
                            ELSE 3
                        END,                               -- Tertiary sort: rating tier
                        average_rating DESC,               -- Final sort: specific rating
                        num_ratings DESC                   -- Then by number of ratings
                    """,
                    user_id,
                    tutor_ids_str,
                )

                # Process recommendations with stronger interest matching
                matched_recommendations = []
                partial_matched_recommendations = []
                other_recommendations = []

                # Track subjects to avoid duplicates while maintaining best rated tutors per subject
                subject_seen = set()

                for tutor in recommended_tutors:
                    subject = tutor["subject"].lower()

                    # Skip if we already have a tutor for this subject
                    if subject in subject_seen:
                        continue

                    recommendation = {
                        "id": str(tutor["id"]),
                        "email": tutor["email"],
                        "subject": tutor["subject"],
                        "specialization": tutor["specialization"],
                        "teaching_style": tutor["teaching_style"],
                        "hourly_rate": float(tutor["hourly_rate"]),
                        "average_rating": float(tutor["average_rating"]),
                        "num_ratings": int(tutor["num_ratings"]),
                        "match_reasons": self._get_match_reasons(
                            user["interest"],
                            user["learning_style"],
                            tutor["subject"],
                            tutor["teaching_style"],
                            tutor["average_rating"],
                        ),
                    }

                    if tutor["exact_match"]:
                        matched_recommendations.append(recommendation)
                        subject_seen.add(subject)
                    elif tutor["partial_match"] and subject not in subject_seen:
                        partial_matched_recommendations.append(recommendation)
                        subject_seen.add(subject)
                    elif subject not in subject_seen:
                        other_recommendations.append(recommendation)
                        subject_seen.add(subject)

                # Combine recommendations prioritizing subject matches
                final_recommendations = (
                    matched_recommendations  # Exact subject matches first
                    + partial_matched_recommendations  # Related subjects second
                    + other_recommendations  # Other subjects last
                )[:top_k]

                response = {
                    "user": {
                        "id": str(user["id"]),
                        "email": user["email"],
                        "interests": user["interest"],
                        "learning_style": user["learning_style"],
                        "city": user["city"],
                    },
                    "recommendations": final_recommendations,
                }

                return response

        except ValueError as e:
            logging.error(f"User not found: {user_id}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    def _get_match_reasons(
        self,
        user_interests: List[str],
        user_style: str,
        tutor_subject: str,
        tutor_style: str,
        rating: float,
    ) -> List[str]:
        reasons = []

        # Ensure user_interests is a list
        if isinstance(user_interests, str):
            try:
                user_interests = json.loads(user_interests)
            except:
                user_interests = [user_interests]
        elif user_interests is None:
            user_interests = []

        # Subject match with more detailed explanation
        if any(
            interest.lower() == tutor_subject.lower() for interest in user_interests
        ):
            reasons.append(
                f"Perfect match: Teaches {tutor_subject} (exactly matches your interest)"
            )
        elif any(
            tutor_subject.lower() in interest.lower()
            or interest.lower() in tutor_subject.lower()
            for interest in user_interests
        ):
            matching_interests = [
                interest
                for interest in user_interests
                if tutor_subject.lower() in interest.lower()
                or interest.lower() in tutor_subject.lower()
            ]
            reasons.append(
                f"Related to your interest in {', '.join(matching_interests)}"
            )

        # Learning style match
        if user_style and tutor_style:
            if user_style == tutor_style:
                reasons.append(
                    f"Teaching style ({tutor_style}) perfectly matches your learning preference"
                )
            elif tutor_style == "flexible":
                reasons.append("Flexible teaching style that can adapt to your needs")
            else:
                reasons.append(
                    f"Different teaching style ({tutor_style}) might provide new perspectives"
                )

        # Rating-based reason
        if rating >= 4.5:
            reasons.append(f"Highly rated tutor (★ {rating:.1f})")
        elif rating >= 4.0:
            reasons.append(f"Well-rated tutor (★ {rating:.1f})")
        elif rating >= 3.5:
            reasons.append(f"Above average rating (★ {rating:.1f})")

        return reasons

    async def update_model(self):
        if self.is_updating:
            return

        self.is_updating = True
        try:
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

                # Fetch tutors with services and ratings with proper type casting
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

        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            self._model_ready = False
            raise
        finally:
            self.is_updating = False

    async def record_interaction(
        self,
        user_id: str,
        tutor_id: str,
        service_id: str,
        rating: Optional[float] = None,
    ):
        try:
            # Store in PostgreSQL
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO session_rating
                    (user_id, service_id, session_rating, created_at)
                    VALUES ($1, $2, $3, $4)
                """,
                    user_id,
                    service_id,
                    rating,
                    datetime.utcnow(),
                )

            # Update cache
            cache_key = f"recommendations:{user_id}"
            await self.redis.delete(cache_key)

        except Exception as e:
            logging.error(f"Error recording interaction: {str(e)}")
            raise

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

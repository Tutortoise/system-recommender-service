import asyncio
from typing import List, Dict
import logging
from uuid import UUID
from fastapi import HTTPException
import sys
from pathlib import Path
from datetime import datetime
from google.cloud import firestore

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MODEL_UPDATE_INTERVAL  # Import settings
from config import initialize_firestore
from .feature_processor import FeatureProcessor
from .recommender import Recommender

class RecommenderService:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.recommender = Recommender()
        self.db = initialize_firestore()
        self.is_updating = False
        self._model_ready = False
        self._lock = asyncio.Lock()

    async def initialize(self):
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
            # Get user data from Firestore
            user_doc = self.db.collection('users').document(user_id).get()

            if not user_doc.exists:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "USER_NOT_FOUND",
                        "message": f"User with ID {user_id} not found",
                        "user_id": user_id,
                    },
                )

            user_data = user_doc.to_dict()

            # Process user features
            user_vector = self.feature_processor.process_user_features({
                "interest": user_data.get('interests', []),
                "learning_style": user_data.get('learning_style'),
                "city": user_data.get('city')
            })

            # Update user features and get recommendations
            self.recommender.update_single_user(user_id, user_vector)
            tutor_ids = self.recommender.get_recommendations(user_id, top_k)

            # Get recommended tutors
            recommendations = []
            for tutor_id in tutor_ids:
                # Get tutor data
                tutor_doc = self.db.collection('tutors').document(tutor_id).get()
                if not tutor_doc.exists:
                    continue

                tutor_data = tutor_doc.to_dict()

                # Get tutor's services
                services = self.db.collection('tutor_services')\
                    .where('tutor_id', '==', tutor_id)\
                    .stream()

                for service in services:
                    service_data = service.to_dict()

                    # Get ratings for this service
                    ratings_query = self.db.collection('session_ratings')\
                        .where('service_id', '==', service.id)\
                        .stream()

                    ratings = [r.to_dict() for r in ratings_query]
                    avg_rating = sum(r['rating'] for r in ratings) / len(ratings) if ratings else 0
                    num_ratings = len(ratings)

                    # Generate match reasons
                    match_reasons = self._get_match_reasons(
                        user_data.get('interests', []),
                        user_data.get('learning_style'),
                        service_data.get('subject'),
                        service_data.get('teaching_style'),
                        avg_rating
                    )

                    if not match_reasons:
                        continue

                    # Add rating-based reason if applicable
                    if avg_rating >= 4.5 and num_ratings >= 10:
                        match_reasons.append(
                            f"Highly rated tutor ({avg_rating:.1f}/5 from {num_ratings} reviews)"
                        )
                    elif avg_rating >= 4.0 and num_ratings >= 5:
                        match_reasons.append(
                            f"Well-rated tutor ({avg_rating:.1f}/5)"
                        )

                    recommendation = {
                        "id": tutor_id,
                        "service_id": service.id,
                        "email": tutor_data.get('email'),
                        "subject": service_data.get('subject'),
                        "specialization": service_data.get('specialization', []),
                        "teaching_style": service_data.get('teaching_style'),
                        "hourly_rate": float(service_data.get('hourly_rate', 0)),
                        "average_rating": float(avg_rating),
                        "num_ratings": num_ratings,
                        "match_reasons": match_reasons,
                        "match_score": len(match_reasons),  # Simple scoring based on number of reasons
                    }

                    recommendations.append(recommendation)

            # Sort recommendations
            recommendations.sort(
                key=lambda x: (
                    -x["match_score"],     # Higher match score first
                    -x["average_rating"],  # Higher rating next
                    -x["num_ratings"]      # More ratings next
                )
            )

            response = {
                "user": {
                    "id": user_id,
                    "email": user_data.get('email'),
                    "interests": user_data.get('interests', []),
                    "learning_style": user_data.get('learning_style'),
                    "city": user_data.get('city'),
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

            # First collect all texts for fitting
            all_texts = []
            users_data = []
            tutors_data = []

            # Get all users
            users = self.db.collection('users').stream()
            for user in users:
                user_data = user.to_dict()
                user_data['id'] = user.id
                users_data.append(user_data)

                interests = user_data.get('interests', [])
                if interests:
                    all_texts.append(" ".join(interests))

            # Get all tutors and their services
            tutors = self.db.collection('tutors').stream()
            for tutor in tutors:
                tutor_data = tutor.to_dict()
                tutor_data['id'] = tutor.id

                # Get tutor's services
                services = self.db.collection('tutor_services')\
                    .where('tutor_id', '==', tutor.id)\
                    .stream()

                for service in services:
                    service_data = service.to_dict()
                    service_data['id'] = service.id

                    # Add texts for fitting
                    if service_data.get('specialization'):
                        all_texts.append(" ".join(service_data['specialization']))
                    if service_data.get('subject'):
                        all_texts.append(service_data['subject'])

                    # Get ratings
                    ratings = self.db.collection('session_ratings')\
                        .where('service_id', '==', service.id)\
                        .stream()

                    ratings_list = list(ratings)
                    rating_data = {
                        "average_rating": sum(r.to_dict()['rating'] for r in ratings_list) / len(ratings_list) if ratings_list else 0,
                        "num_ratings": len(ratings_list)
                    }

                    tutors_data.append({
                        'tutor': tutor_data,
                        'service': service_data,
                        'ratings': rating_data
                    })

            # Fit feature processor first
            self.feature_processor.fit(all_texts)

            # Now process features after fitting
            user_features = {}
            tutor_features = {}

            # Process user features
            for user_data in users_data:
                user_features[user_data['id']] = self.feature_processor.process_user_features({
                    "interest": user_data.get('interests', []),
                    "learning_style": user_data.get('learning_style'),
                    "city": user_data.get('city')
                })

            # Process tutor features
            for tutor_data in tutors_data:
                location = (
                    (float(tutor_data['tutor'].get('latitude')), float(tutor_data['tutor'].get('longitude')))
                    if tutor_data['tutor'].get('latitude') and tutor_data['tutor'].get('longitude')
                    else None
                )

                tutor_features[tutor_data['tutor']['id']] = self.feature_processor.process_tutor_features(
                    {"location": location},
                    tutor_data['service'],
                    tutor_data['ratings']
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
                await asyncio.sleep(MODEL_UPDATE_INTERVAL)

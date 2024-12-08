import numpy as np
from typing import List, Dict
import logging
from .subject_embeddings import SubjectEmbedding


class FeatureProcessor:
    def __init__(self):
        self.is_fitted = False
        self.feature_dim = None
        self.vector_size = 50
        self.subject_embedding = SubjectEmbedding(vector_size=self.vector_size)

    def fit(self, texts: List[str]):
        """Initialize feature processor dimensions"""
        if not texts:
            raise ValueError("No texts provided for fitting")

        self.is_fitted = True
        self.feature_dim = (
            self.vector_size  # subject/methodology embeddings
            + 3  # learning style
            + 2  # location (city, district)
            + 1  # price normalization
            + 3  # lesson type
            + 2  # ratings (avg_rating, completion_rate)
        )

    def _get_text_vector(self, text: str) -> np.ndarray:
        """Get averaged word vectors for text"""
        if not text:
            return np.zeros(self.vector_size)

        words = text.lower().split()
        vectors = []

        for word in words:
            try:
                vectors.append(self.subject_embedding.get_vector(word))
            except KeyError:
                continue

        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def process_user_features(self, user_data: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Feature processor not fitted")

        try:
            # Process interests
            interest_vector = np.zeros(self.vector_size)
            if user_data.get("interests"):
                vectors = []
                for interest in user_data["interests"]:
                    vectors.append(self._get_text_vector(interest))
                if vectors:
                    interest_vector = np.mean(vectors, axis=0)

            # Process learning style
            learning_style_vec = self._encode_learning_style(
                user_data.get("learning_style", "")
            )

            # Process location
            location_vec = self._encode_location(
                user_data.get("city", ""), user_data.get("district", "")
            )

            return np.concatenate(
                [
                    interest_vector * 2.0,  # Highest weight for interests
                    learning_style_vec * 0.5,  # Learning style importance
                    location_vec * 0.3,  # Location importance
                ]
            )

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def process_tutor_features(
        self, tutor_data: Dict, tutory_data: Dict, rating_data: Dict = None
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Feature processor not fitted")

        try:
            # Process subject vector
            subject_vector = self._get_text_vector(tutory_data.get("subject", ""))

            # Process methodology vector
            methodology_vector = self._get_text_vector(
                tutory_data.get("teaching_methodology", "")
            )

            # Process location
            location_vec = self._encode_location(
                tutor_data.get("city", ""), tutor_data.get("district", "")
            )

            # Process lesson type
            lesson_type_vec = self._encode_lesson_type(
                tutory_data.get("type_lesson", "")
            )

            # Process price normalization (50k-200k range)
            price_vec = self._encode_price(tutory_data.get("hourly_rate", 0))

            # Process rating stats with default values if rating_data is None
            rating_data = rating_data or {}
            stats_vec = np.array(
                [
                    rating_data.get("avg_rating", 0) / 5.0,  # Normalize rating to 0-1
                    rating_data.get("completion_rate", 0),  # Already 0-1
                ]
            )

            return np.concatenate(
                [
                    subject_vector * 2.0,  # Highest weight for subject matching
                    methodology_vector * 0.8,  # Teaching methodology importance
                    location_vec * 0.5,  # Location importance
                    lesson_type_vec * 0.3,  # Lesson type importance
                    [price_vec * 0.4],  # Price importance
                    stats_vec * 0.6,  # Stats importance
                ]
            )

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def _encode_learning_style(self, style: str) -> np.ndarray:
        styles = {
            "visual": [1, 0, 0],
            "auditory": [0, 1, 0],
            "kinesthetic": [0, 0, 1],
        }
        return np.array(styles.get(style.lower(), [0, 0, 0]))

    def _encode_lesson_type(self, lesson_type: str) -> np.ndarray:
        types = {
            "online": [1, 0, 0],
            "offline": [0, 1, 0],
            "both": [0, 0, 1],
        }
        return np.array(types.get(lesson_type.lower(), [0, 0, 0]))

    def _encode_location(self, city: str, district: str) -> np.ndarray:
        if not city or not district:
            return np.zeros(2)

        city_hash = hash(city.lower()) % 1000 / 1000
        district_hash = hash(district.lower()) % 1000 / 1000
        return np.array([city_hash, district_hash])

    def _encode_price(self, hourly_rate: int) -> float:
        # Normalize price to 0-1 range (assuming max price is 1000)
        return min(float(hourly_rate) / 10_000_000.0, 1.0)

    def _get_match_reasons(self, learner: Dict, tutor: Dict, tutory: Dict) -> List[str]:
        reasons = []

        # Check learning style match
        if learner.get("learning_style") == tutor.get("learning_style"):
            reasons.append("Learning style matches your preference")

        # Check location match
        if learner.get("city") == tutor.get("city"):
            if learner.get("district") == tutor.get("district"):
                reasons.append("Located in your district")
            else:
                reasons.append("Located in your city")

        # Check price reasonability
        hourly_rate = tutory.get("hourly_rate", 0)
        if hourly_rate <= 50:
            reasons.append("Affordable pricing")

        # Check lesson type preference
        if tutory.get("type_lesson") == "both":
            reasons.append("Flexible lesson format (online/offline)")

        # Add category match reason
        if tutory.get("category_name"):  # Changed from subject to category_name
            reasons.append(f"Teaches {tutory['category_name']}")

        return reasons

from gensim.models import Word2Vec
import numpy as np
from typing import List, Dict
import logging
from .subject_embeddings import SubjectEmbedding


class FeatureProcessor:
    def __init__(self):
        self.is_fitted = False
        self.feature_dim = None
        self.vector_size = 300  # Match Google's Word2Vec dimensions
        self.subject_embedding = SubjectEmbedding(vector_size=self.vector_size)

    def fit(self, texts: List[str]):
        """Initialize feature processor dimensions"""
        if not texts:
            raise ValueError("No texts provided for fitting")

        # Just set dimensions and mark as fitted
        self.is_fitted = True
        self.feature_dim = (
            self.vector_size + 3 + 1
        )  # word2vec + learning_style + city + price

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
            raise ValueError("Word2Vec model not fitted")

        try:
            # Process interests
            interests = user_data.get("interest", [])
            if isinstance(interests, str):
                interests = [interests]

            # Get averaged vectors for all interests
            interest_vectors = []
            for interest in interests:
                vec = self._get_text_vector(interest)
                interest_vectors.append(vec)

            if interest_vectors:
                text_features = np.mean(interest_vectors, axis=0)
            else:
                text_features = np.zeros(self.vector_size)

            # Encode other features
            learning_style = self._encode_learning_style(
                user_data.get("learning_style", "")
            )
            city_encoding = self._encode_city(user_data.get("city", ""))

            return np.concatenate(
                [
                    text_features * 2.0,  # Higher weight for subject matching
                    learning_style * 0.5,
                    [city_encoding * 0.3],
                ]
            )

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def process_tutor_features(
        self, tutor_data: Dict, service_data: Dict, rating_data: Dict = None
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Word2Vec model not fitted")

        try:
            # Process subject and specialization
            subject = service_data.get("subject", "")
            specialization = service_data.get("specialization", [])

            # Get subject vector
            subject_vector = self._get_text_vector(subject)

            # Get specialization vectors
            spec_vectors = []
            for spec in specialization:
                vec = self._get_text_vector(spec)
                spec_vectors.append(vec)

            # Combine vectors with weights
            if spec_vectors:
                spec_vector = np.mean(spec_vectors, axis=0)
                text_features = np.mean(
                    [
                        subject_vector * 2.0,  # Higher weight for main subject
                        spec_vector * 1.0,
                    ],
                    axis=0,
                )
            else:
                text_features = subject_vector * 2.0

            # Process other features
            teaching_style = self._encode_teaching_style(
                service_data.get("teaching_style", "")
            )
            location_encoding = self._encode_location(tutor_data.get("location", None))
            price_encoding = self._encode_price(service_data.get("hourly_rate", 0))
            rating_feature = self._encode_rating(rating_data) if rating_data else 0.0

            return np.concatenate(
                [
                    text_features * 3.0,  # Highest weight for subject matching
                    teaching_style * 0.5,
                    [location_encoding * 0.3],
                    [price_encoding * 0.2],
                    [rating_feature * 0.4],
                ]
            )

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def _encode_learning_style(self, style: str) -> np.ndarray:
        styles = {
            "structured": [1, 0, 0],
            "flexible": [0, 1, 0],
            "project_based": [0, 0, 1],
        }
        return np.array(styles.get(style, [0, 0, 0]))

    def _encode_teaching_style(self, style: str) -> np.ndarray:
        styles = {
            "structured": [1, 0, 0],
            "flexible": [0, 1, 0],
            "project_based": [0, 0, 1],
        }
        return np.array(styles.get(style, [0, 0, 0]))

    def _encode_city(self, city: str) -> float:
        if not city:
            return 0.0
        return (hash(city) % 1000) / 1000.0

    def _encode_location(self, location: tuple) -> float:
        """Encode location as a single float value."""
        if (
            not location
            or not isinstance(location, (tuple, list))
            or len(location) != 2
        ):
            return 0.0
        try:
            lat, lon = map(float, location)
            normalized_lat = (lat + 90) / 180  # Convert [-90, 90] to [0, 1]
            normalized_lon = (lon + 180) / 360  # Convert [-180, 180] to [0, 1]
            # Combine into single value
            return (normalized_lat + normalized_lon) / 2
        except (ValueError, TypeError):
            return 0.0

    def _encode_price(self, hourly_rate: int) -> float:
        return min(hourly_rate / 1000.0, 1.0)

    def _encode_rating(self, rating_data: Dict) -> float:
        """Encode rating information into a feature"""
        if not rating_data:
            return 0.0

        avg_rating = rating_data.get("average_rating", 0)
        num_ratings = rating_data.get("num_ratings", 0)

        # Weighted rating calculation (similar to IMDB weighted rating)
        C = 10  # minimum number of ratings required for weight
        m = 3.0  # minimum rating threshold

        weighted_rating = (C * m + num_ratings * avg_rating) / (C + num_ratings)
        return min(weighted_rating / 5.0, 1.0)  # Normalize to [0,1]

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
            if best_sim > 0.85:  # Perfect match (exact or very close subjects)
                reasons.append(
                    f"Perfect match: Teaches {tutor_subject} (exactly matches your interest in {best_match})"
                )
            elif best_sim > 0.75:  # Strong match (closely related subjects)
                reasons.append(
                    f"Strong match: {tutor_subject} is closely related to your interest in {best_match}"
                )
            elif best_sim > 0.65:  # Related match (somewhat related subjects)
                reasons.append(f"Related to your interest in {best_match}")
            else:
                return []  # Don't include unrelated subjects

        return reasons

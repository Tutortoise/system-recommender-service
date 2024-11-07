import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import logging


class FeatureProcessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.is_fitted = False
        self.feature_dim = None
        self.rating_weight = 0.3  # Weight for rating influence

    def fit(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided for fitting")

        self.tfidf.fit(texts)
        self.is_fitted = True
        # TF-IDF features + learning_style + city encoding + price sensitivity
        self.feature_dim = self.tfidf.max_features + 3 + 1 + 1

    def process_user_features(self, user_data: Dict) -> np.ndarray:
        try:
            if not self.is_fitted:
                raise ValueError("TF-IDF not fitted")

            # Process interests
            interests = " ".join(user_data.get("interest", []))
            if not interests.strip():
                text_features = np.zeros(self.tfidf.max_features)
            else:
                text_features = self.tfidf.transform([interests]).toarray()[0]

            learning_style = self._encode_learning_style(
                user_data.get("learning_style", "")
            )

            city_encoding = self._encode_city(user_data.get("city", ""))

            return np.concatenate([text_features, learning_style, [city_encoding]])

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def process_tutor_features(
        self, tutor_data: Dict, service_data: Dict, rating_data: Dict = None
    ) -> np.ndarray:
        try:
            if not self.is_fitted:
                raise ValueError("TF-IDF not fitted")

            # Significantly increase the weight of subject matching
            subject_weight = 5.0

            # Process subject with higher weight than specialization
            subject = service_data.get("subject", "")
            specialization = " ".join(service_data.get("specialization", []))

            # Give more weight to subject than specialization
            expertise = f"{subject} {subject} {subject} {specialization}"  # Repeat subject to increase its weight

            if not expertise.strip():
                text_features = np.zeros(self.tfidf.max_features)
            else:
                text_features = (
                    self.tfidf.transform([expertise]).toarray()[0] * subject_weight
                )

            # Enhanced features with adjusted weights
            teaching_style = self._encode_teaching_style(
                service_data.get("teaching_style", "")
            )
            location_encoding = self._encode_location(tutor_data.get("location", None))
            price_encoding = self._encode_price(service_data.get("hourly_rate", 0))
            rating_feature = self._encode_rating(rating_data) if rating_data else 0.0

            # Combine features with adjusted weights
            return np.concatenate(
                [
                    text_features
                    * 3.0,  # Highest weight for subject/expertise matching
                    teaching_style * 0.5,  # Reduced weight for teaching style
                    [location_encoding * 0.3],  # Reduced weight for location
                    [price_encoding * 0.2],  # Reduced weight for price
                    [rating_feature * 0.4],  # Reduced weight for rating
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

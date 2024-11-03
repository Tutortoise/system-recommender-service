import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import logging


class FeatureProcessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.is_fitted = False
        self.feature_dim = None

    def fit(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided for fitting")

        self.tfidf.fit(texts)
        self.is_fitted = True
        # Calculate total feature dimension
        self.feature_dim = self.tfidf.max_features + 9  # TF-IDF features + categorical

    def process_user_features(self, user_data: Dict) -> np.ndarray:
        try:
            if not self.is_fitted:
                raise ValueError("TF-IDF not fitted")

            # Process text features
            interests = " ".join(user_data.get("interests", []))
            if not interests.strip():
                text_features = np.zeros(self.tfidf.max_features)
            else:
                text_features = self.tfidf.transform([interests]).toarray()[0]

            # Process categorical features
            categorical = self._process_user_categorical(
                user_data.get("learning_style", ""),
                user_data.get("experience_level", ""),
            )

            return np.concatenate([text_features, categorical])

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def process_mentor_features(self, mentor_data: Dict) -> np.ndarray:
        try:
            if not self.is_fitted:
                raise ValueError("TF-IDF not fitted")

            # Process text features
            expertise = " ".join(mentor_data.get("expertise", []))
            if not expertise.strip():
                text_features = np.zeros(self.tfidf.max_features)
            else:
                text_features = self.tfidf.transform([expertise]).toarray()[0]

            # Process categorical features
            categorical = self._process_mentor_categorical(
                mentor_data.get("teaching_style", ""),
                mentor_data.get("years_experience", 0)
            )

            return np.concatenate([text_features, categorical])

        except Exception as e:
            logging.error(f"Feature processing error: {str(e)}", exc_info=True)
            raise

    def _process_user_categorical(self, learning_style: str, experience_level: str) -> np.ndarray:
        categorical = np.zeros(9)  # 3 learning styles + 3 experience levels + 3 teaching styles

        # Learning style
        style_map = {"visual": 0, "auditory": 1, "kinesthetic": 2}
        if learning_style in style_map:
            categorical[style_map[learning_style]] = 1

        # Experience level
        level_map = {"beginner": 3, "intermediate": 4, "advanced": 5}
        if experience_level in level_map:
            categorical[level_map[experience_level]] = 1

        return categorical

    def _process_mentor_categorical(self, teaching_style: str, years_experience: int) -> np.ndarray:
        categorical = np.zeros(9)  # Same dimension as user features for compatibility

        # Teaching style
        style_map = {"structured": 6, "flexible": 7, "project_based": 8}
        if teaching_style in style_map:
            categorical[style_map[teaching_style]] = 1

        # Normalize years of experience to a value between 0 and 1
        exp_value = min(years_experience / 20.0, 1.0)  # Cap at 20 years
        categorical[3:6] = exp_value  # Apply to experience level slots

        return categorical

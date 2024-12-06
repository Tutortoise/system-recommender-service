from vowpalwabbit import pyvw
import numpy as np
from typing import List, Dict
import logging
from collections import defaultdict

from recommender.interaction import InteractionTracker


class Recommender:
    def __init__(self):
        """
        Initialize VW recommender
        Args:
            cb_type: Type of contextual bandit ('ips' or 'dr')
            exploration_rate: Epsilon for epsilon-greedy exploration
        """
        self.vw = pyvw.vw(
            """
            --cb_explore_adf
            --quiet
            --passes 1
            --cache_file cache.vw
            --loss_function logistic
            --learning_rate 0.1
            --power_t 0.5
            --random_seed 0
        """
        )
        self.user_features = {}
        self.mentor_features = {}
        self.mentor_ids = []
        self.similarity_cache = {}
        self.feature_importance = defaultdict(float)
        self.processed_mentor_features = {}
        self._last_predictions = {}
        self.interaction_tracker = InteractionTracker()

        self.user_feature_dim = None
        self.mentor_feature_dim = None

    def update_single_user(self, user_id: str, user_vector: np.ndarray):
        """Update features for a single user and clear their cache entries."""
        try:
            self.user_features[user_id] = user_vector
            processed_features = self._process_features(user_vector, is_user=True)

            cache_keys = [
                k for k in self.similarity_cache if k.startswith(f"{user_id}:")
            ]
            for key in cache_keys:
                self.similarity_cache.pop(key)
            if user_id in self._last_predictions:
                del self._last_predictions[user_id]

            return processed_features

        except Exception as e:
            logging.error(f"Error updating user features: {str(e)}")
            raise

    def process_interaction(
        self,
        user_id: str,
        tutory_id: str,
        tutor_id: str,
        category_id: str,
        category_name: str,
    ):
        """Process user click on a tutory"""
        try:
            # Track interaction and get weight
            weight = self.interaction_tracker.add_interaction(
                user_id=user_id,
                tutory_id=tutory_id,
                tutor_id=tutor_id,
                category_id=category_id,
                category_name=category_name,
            )

            # Use click as implicit feedback
            if (
                user_id in self.user_features
                and tutor_id in self.processed_mentor_features
            ):
                user_processed = self._process_features(
                    self.user_features[user_id], is_user=True
                )

                example = self._create_example(
                    user_processed, self.processed_mentor_features[tutor_id]
                )
                example.append(f"{weight}:1.0")
                self.vw.learn(example)

                # Clear user's cache
                self._clear_user_caches(user_id)

                # Update user preferences
                self._update_user_preferences(user_id)

            return True

        except Exception as e:
            logging.error(f"Error processing click interaction: {str(e)}")
            return False

    def _format_features(self, features: Dict) -> List[str]:
        """Convert features to VW format"""
        feature_strings = []

        for namespace, values in features.items():
            if isinstance(values, (np.ndarray, list)):
                feature_parts = []
                for i, v in enumerate(values):
                    if abs(v) > 1e-6:
                        feature_parts.append(f"{namespace}_{i}:{v:.6f}")
                if feature_parts:
                    feature_strings.extend(feature_parts)
            elif isinstance(values, (int, float)):
                if abs(values) > 1e-6:
                    feature_strings.append(f"{namespace}:{values:.6f}")

        return feature_strings

    def _create_example(self, user_features: Dict, mentor_features: Dict) -> List[str]:
        """Create a VW example in the shared/action format"""
        user_namespaces = self._format_features(user_features)
        mentor_namespaces = self._format_features(mentor_features)

        # Format the example properly for VW
        example = []

        # Add shared features
        shared_features = " ".join(user_namespaces)
        example.append(f"shared |u {shared_features}")

        # Add action features
        action_features = " ".join(mentor_namespaces)
        example.append(f"|a {action_features}")

        return example

    def build_index(self, mentor_features: Dict[str, np.ndarray]):
        """Build the mentor index"""
        if not mentor_features:
            raise ValueError("No mentor features provided")

        self.mentor_features = mentor_features
        self.mentor_ids = list(mentor_features.keys())
        self.processed_mentor_features.clear()

        # Process first mentor to get feature dimensions
        first_mentor = next(iter(mentor_features.values()))
        self._process_features(first_mentor, is_user=False)

        # Pre-process all mentor features
        for mid, features in mentor_features.items():
            try:
                self.processed_mentor_features[mid] = self._process_features(
                    features, is_user=False
                )
            except Exception as e:
                logging.error(f"Error processing features for mentor {mid}: {str(e)}")
                continue

    def _process_features(self, features: np.ndarray, is_user: bool = False) -> Dict:
        """
        Convert numpy array to named features dynamically based on dimensions
        Args:
            features: numpy array of features
            is_user: True if processing user features, False if mentor features
        """
        try:
            feature_dict = {}
            start = 0

            # Set feature dimensions if not set
            if is_user and self.user_feature_dim is None:
                self.user_feature_dim = len(features)
            elif not is_user and self.mentor_feature_dim is None:
                self.mentor_feature_dim = len(features)

            step = 50
            for i in range(0, len(features), step):
                namespace = f"f{i//step}"
                end = min(i + step, len(features))
                feature_dict[namespace] = features[i:end]

            return feature_dict

        except Exception as e:
            logging.error(
                f"Error processing {'user' if is_user else 'mentor'} features: {str(e)}, "
                f"shape: {features.shape}"
            )
            raise

    def get_recommendations(self, user_id: str, top_k: int = 5) -> List[str]:
        """Get recommendations using VW contextual bandit"""
        if user_id not in self.user_features:
            raise ValueError(f"User {user_id} not found")

        cache_key = f"{user_id}:{top_k}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        try:
            user_processed = self._process_features(
                self.user_features[user_id], is_user=True
            )
            scores = []

            for mentor_id in self.mentor_ids:
                example = self._create_example(
                    user_processed, self.processed_mentor_features[mentor_id]
                )

                # VW returns a list of probabilities for each action
                pred = self.vw.predict(example)
                # Use the first probability as the score
                score = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
                scores.append((mentor_id, float(score)))

            self._last_predictions[user_id] = dict(scores)
            scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [mid for mid, _ in scores[:top_k]]

            self.similarity_cache[cache_key] = recommendations
            return recommendations

        except Exception as e:
            logging.error(
                f"Error getting recommendations: {str(e)}, pred type: {type(pred)}"
            )
            raise

    def learn(self, user_id: str, mentor_id: str, reward: float):
        """Online learning from user feedback"""
        if user_id not in self.user_features or mentor_id not in self.mentor_features:
            return

        try:
            user_processed = self._process_features(
                self.user_features[user_id], is_user=True
            )
            example = self._create_example(
                user_processed, self.processed_mentor_features[mentor_id]
            )

            # Get probability from last prediction
            prob = self._last_predictions.get(user_id, {}).get(mentor_id, 0.5)

            # Create the learning example with cost
            learn_example = example.copy()
            learn_example.append(f"{-reward}:{prob}")

            # Learn from this example
            self.vw.learn(learn_example)

            # Clear caches
            if user_id in self._last_predictions:
                del self._last_predictions[user_id]
            cache_keys = [
                k for k in self.similarity_cache if k.startswith(f"{user_id}:")
            ]
            for key in cache_keys:
                self.similarity_cache.pop(key)

        except Exception as e:
            logging.error(f"Error during learning: {str(e)}")

    def reset_caches(self):
        """Reset the VW model and caches"""
        self.vw = pyvw.vw("--cb_explore_adf --quiet")
        self.similarity_cache.clear()
        self.feature_importance.clear()
        self.processed_mentor_features.clear()
        self._last_predictions.clear()

    def _update_user_preferences(self, user_id: str):
        """Update user preferences based on interactions"""
        category_prefs = self.interaction_tracker.get_category_preferences(user_id)
        if not category_prefs:
            return

        if user_id in self.user_features:
            self.user_features[user_id] = np.concatenate(
                [self.user_features[user_id], np.array(list(category_prefs.values()))]
            )

    def update_user_features(self, user_features: Dict[str, np.ndarray]):
        """Update user features and clear related caches"""
        self.user_features = user_features
        self.similarity_cache.clear()
        self._last_predictions.clear()

    def __del__(self):
        """Cleanup VW instance"""
        if hasattr(self, "vw"):
            self.vw.finish()

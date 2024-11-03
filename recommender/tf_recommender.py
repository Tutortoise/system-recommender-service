import tensorflow as tf
import numpy as np
from typing import List, Dict
import logging


class TensorFlowRecommender:
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.user_features = {}
        self.mentor_features = {}
        self.mentor_ids = []
        self.similarity_cache = {}

        # TF Models
        self.user_encoder = self._build_encoder("user")
        self.mentor_encoder = self._build_encoder("mentor")

        # Cached tensors
        self.mentor_embeddings = None

    def _build_encoder(self, name: str):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu", name=f"{name}_dense1"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu", name=f"{name}_dense2"),
                tf.keras.layers.Dense(self.embedding_dim, name=f"{name}_output"),
                tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1)),
            ]
        )

    @tf.function(reduce_retracing=True)
    def _compute_similarities(self, user_embedding, mentor_embeddings):
        # Reshape user embedding to [1, embedding_dim]
        user_embedding = tf.reshape(user_embedding, [1, -1])
        # Compute cosine similarity using matrix multiplication
        similarities = tf.matmul(user_embedding, mentor_embeddings, transpose_b=True)
        return tf.squeeze(similarities)

    def build_index(self, mentor_features: Dict[str, np.ndarray]):
        if not mentor_features:
            raise ValueError("No mentor features provided")

        self.mentor_features = mentor_features
        self.mentor_ids = list(mentor_features.keys())

        # Convert to TF tensor and compute embeddings
        feature_matrix = tf.convert_to_tensor(
            list(mentor_features.values()), dtype=tf.float32
        )
        self.mentor_embeddings = self.mentor_encoder(feature_matrix)

    def update_user_features(self, user_features: Dict[str, np.ndarray]):
        self.user_features = user_features
        self.similarity_cache = {}

    def get_recommendations(self, user_id: str, top_k: int = 5) -> List[str]:
        if user_id not in self.user_features:
            raise ValueError(f"User {user_id} not found")

        cache_key = f"{user_id}:{top_k}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        try:
            # Get user embedding
            user_vector = tf.convert_to_tensor(
                self.user_features[user_id], dtype=tf.float32
            )
            user_embedding = self.user_encoder(tf.expand_dims(user_vector, 0))[0]

            # Compute similarities
            similarities = self._compute_similarities(
                user_embedding, self.mentor_embeddings
            )

            # Get top-k indices
            _, indices = tf.math.top_k(similarities, k=min(top_k, len(self.mentor_ids)))
            recommendations = [self.mentor_ids[i] for i in indices.numpy()]

            # Cache results
            self.similarity_cache[cache_key] = recommendations
            return recommendations

        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            raise

    def update_single_user(self, user_id: str, user_vector: np.ndarray):
        """Update features for a single user and clear their cache entries."""
        self.user_features[user_id] = user_vector
        # Clear only this user's cache entries
        cache_keys = [k for k in self.similarity_cache if k.startswith(f"{user_id}:")]
        for key in cache_keys:
            self.similarity_cache.pop(key)

import numpy as np
from typing import List
import logging
import gensim.downloader as api


class SubjectEmbedding:
    def __init__(self, vector_size=300):
        self.vector_size = vector_size
        self.model = None
        self.subject_vectors = {}

        try:
            logging.info("Loading Word2Vec model... This might take a few minutes the first time.")
            self.model = api.load("word2vec-google-news-300")
            logging.info("Word2Vec model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading pre-trained model: {str(e)}")
            raise

    def get_subject_similarity(self, subject1: str, subject2: str) -> float:
        """Calculate similarity between two subjects using pre-trained embeddings"""
        if not self.model:
            return 0.0

        try:
            # Split multi-word subjects and get average similarity
            words1 = subject1.lower().split()
            words2 = subject2.lower().split()

            similarities = []
            for w1 in words1:
                for w2 in words2:
                    try:
                        sim = self.model.similarity(w1, w2)
                        similarities.append(sim)
                    except KeyError:
                        continue

            return max(similarities) if similarities else 0.0

        except Exception as e:
            logging.error(f"Error computing similarity: {str(e)}")
            return 0.0

    def get_vector(self, subject: str) -> np.ndarray:
        """Get averaged vector for a subject using pre-trained embeddings"""
        words = subject.lower().split()
        vectors = []

        for word in words:
            try:
                vectors.append(self.model[word])
            except KeyError:
                continue

        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def get_similar_subjects(self, subject: str, top_k: int = 5) -> List[tuple]:
        """Get similar subjects using pre-trained embeddings"""
        if not self.model:
            return []

        try:
            words = subject.lower().split()
            similar_words = []

            for word in words:
                try:
                    similar = self.model.most_similar(word, topn=top_k)
                    similar_words.extend(similar)
                except KeyError:
                    continue

            # Sort by similarity and take top_k
            similar_words.sort(key=lambda x: x[1], reverse=True)
            return similar_words[:top_k]

        except Exception as e:
            logging.error(f"Error finding similar subjects: {str(e)}")
            return []

import numpy as np
from typing import List
import logging
import os
from pathlib import Path


class SubjectEmbedding:
    def __init__(self, vector_size=50):
        self.vector_size = vector_size
        self.model = None
        self.word_to_index = {}
        self.vectors = None

        try:
            model_dir = Path("recommender/embedding_model")
            vector_file = model_dir / "glove.6B.50d.npy"
            vocab_file = model_dir / "glove.6B.50d.vocab.txt"

            with open(vocab_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    word = line.strip()
                    self.word_to_index[word] = idx

            self.vectors = np.load(vector_file, mmap_mode="r")
            logging.info(
                f"Loaded GloVe embeddings with vocabulary size: {len(self.word_to_index)}"
            )

        except Exception as e:
            logging.error(f"Error loading GloVe embeddings: {str(e)}")
            raise

    def get_vector(self, word: str) -> np.ndarray:
        """Get vector for a word"""
        try:
            idx = self.word_to_index.get(word.lower())
            if idx is not None:
                return self.vectors[idx]
            return np.zeros(self.vector_size)
        except Exception as e:
            logging.error(f"Error getting vector for word '{word}': {str(e)}")
            return np.zeros(self.vector_size)

    def get_subject_similarity(self, subject1: str, subject2: str) -> float:
        """Calculate similarity between two subjects"""
        try:
            words1 = subject1.lower().split()
            words2 = subject2.lower().split()

            vec1 = np.mean([self.get_vector(w) for w in words1], axis=0)
            vec2 = np.mean([self.get_vector(w) for w in words2], axis=0)

            if vec1.size > 0 and vec2.size > 0:
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    return float(np.dot(vec1, vec2) / (norm1 * norm2))
            return 0.0

        except Exception as e:
            logging.error(f"Error computing similarity: {str(e)}")
            return 0.0

    def get_similar_subjects(self, subject: str, top_k: int = 5) -> List[tuple]:
        """Get similar subjects"""
        try:
            subject_vec = np.mean(
                [self.get_vector(w) for w in subject.lower().split()], axis=0
            )

            # Calculate similarities with all words
            similarities = []
            for word, idx in self.word_to_index.items():
                vec = self.vectors[idx]
                sim = np.dot(subject_vec, vec) / (
                    np.linalg.norm(subject_vec) * np.linalg.norm(vec)
                )
                similarities.append((word, float(sim)))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logging.error(f"Error finding similar subjects: {str(e)}")
            return []

    @staticmethod
    def prepare_embeddings(glove_file: str, output_dir: Path):
        """
        Convert GloVe text file to numpy memory-mapped format
        """
        vectors = []
        words = []

        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                vectors.append(vector)
                words.append(word)

        vectors = np.array(vectors)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "glove.6B.50d.npy", vectors)

        with open(output_dir / "glove.6B.50d.vocab.txt", "w", encoding="utf-8") as f:
            for word in words:
                f.write(f"{word}\n")

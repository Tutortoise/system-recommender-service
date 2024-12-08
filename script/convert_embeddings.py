import os
import pathlib
import zipfile
import numpy as np
import wget
import logging
from typing import Optional

def download_glove(download_dir: pathlib.Path) -> pathlib.Path:
    """Download GloVe embeddings if not exists"""
    glove_zip = download_dir / "glove.6B.zip"

    if not glove_zip.exists():
        print("Downloading GloVe embeddings...")
        wget.download("http://nlp.stanford.edu/data/glove.6B.zip", str(glove_zip))
    return glove_zip

def extract_glove(zip_path: pathlib.Path, extract_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Extract GloVe 50d embeddings"""
    with zipfile.ZipFile(zip_path) as zf:
        target_file = "glove.6B.50d.txt"
        zf.extract(target_file, extract_dir)
        return extract_dir / target_file

def convert_to_numpy(glove_file: pathlib.Path, output_dir: pathlib.Path):
    """Convert GloVe text file to numpy format"""
    vectors = []
    words = []

    print("Converting GloVe embeddings to numpy format...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            vectors.append(vector)
            words.append(word)

    vectors = np.array(vectors)

    # Save vectors and vocabulary
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "glove.6B.50d.npy", vectors)

    with open(output_dir / "glove.6B.50d.vocab.txt", "w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")

def setup_embeddings(base_dir: Optional[pathlib.Path] = None):
    try:
        # Use provided base_dir or default to current directory
        base_dir = base_dir or pathlib.Path.cwd()

        # Define directories
        download_dir = base_dir / "downloads"
        extract_dir = base_dir / "extracted"
        embed_dir = base_dir / "recommender" / "embedding_model"

        # Create necessary directories
        for directory in [download_dir, extract_dir, embed_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Check if files already exist
        if (embed_dir / "glove.6B.50d.npy").exists() and \
           (embed_dir / "glove.6B.50d.vocab.txt").exists():
            print("Embedding files already exist. Skipping conversion.")
            return

        # Download and process
        glove_zip = download_glove(download_dir)
        glove_file = extract_glove(glove_zip, extract_dir)
        convert_to_numpy(glove_file, embed_dir)

        # Clean up
        if glove_file and glove_file.exists():
            glove_file.unlink()
        if glove_zip.exists():
            glove_zip.unlink()

        print("\nEmbedding setup completed successfully!")

    except Exception as e:
        logging.error(f"Error setting up embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    setup_embeddings()

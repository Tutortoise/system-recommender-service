# Tutortoise Recommender System

A real-time recommendation system (Online Learning) for tutories based on user interactions and preferences. The system uses a hybrid approach combining collaborative filtering, content-based filtering, and location-based matching to provide personalized recommendations.

## Features

- Real-time recommendation updates based on user interactions
- Text similarity using Word2Vec embeddings
- Location-based matching
- Learning style compatibility
- Dynamic price and rating considerations

## Environment Variables

```env
# Database Settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Service Settings
SERVICE_PORT=8000
SERVICE_HOST=0.0.0.0

# Model Settings
MODEL_UPDATE_INTERVAL=21600  # Model update interval in seconds
INTERACTION_WEIGHT=0.3       # Weight for user interactions
CLEANUP_INTERVAL=3600       # Cleanup interval for old interactions in seconds
```

## API Endpoints

### GET /recommendations/{learner_id}

Get personalized recommendations for a user.

- Query Parameters:
  - `top_k` (int, default=5): Number of recommendations to return
  - `strict` (bool, default=false): Whether to require exact number of recommendations

### GET /interaction/{learner_id}/{tutories_id}

Track user interaction with a tutory.

### POST /model/update

Manually trigger model update.

### POST /model/reset

Reset model state and clear caches.

### GET /model/status

Get current model status and statistics.

### GET /health

Check service health status.

## Running with Docker

Build:

```bash
docker build -t recommender:latest .
```

Run:

```bash
docker run -d \
  --name recommender \
  -p 8000:8000 \
  -e POSTGRES_HOST=host.docker.internal \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  recommender:latest
```

## Dependencies

- FastAPI
- VowpalWabbit
- Gensim
- PostgreSQL (asyncpg)
- NumPy

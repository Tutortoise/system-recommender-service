import os

# Model update interval in seconds (default: 6 hour)
MODEL_UPDATE_INTERVAL = int(os.getenv("MODEL_UPDATE_INTERVAL", 6 * 60 * 60))

# Firestore settings
FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID", "demo-project")
FIRESTORE_EMULATOR_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", "localhost:8081")

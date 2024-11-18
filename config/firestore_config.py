import os
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials

def initialize_firestore():
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        db = firestore.Client(
            project=os.getenv("FIRESTORE_PROJECT_ID", "demo-project"),
            credentials=AnonymousCredentials(),
        )
    else:
        db = firestore.Client()

    return db

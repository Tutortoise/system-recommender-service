from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import initialize_firestore

def test_firestore():
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8081"
    db = initialize_firestore()

    # Try to write some test data
    doc_ref = db.collection('test').document('test_doc')
    doc_ref.set({
        'test': 'data',
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Read it back
    doc = doc_ref.get()
    print("Retrieved data:", doc.to_dict())

if __name__ == "__main__":
    test_firestore()

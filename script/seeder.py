import asyncio
from faker import Faker
import uuid
from datetime import datetime
import random
from typing import List
from google.cloud import firestore
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import initialize_firestore
fake = Faker()

LEARNING_STYLES = ["structured", "flexible", "project-based"]
SUBJECTS = [
    "Mathematics", "Physics", "Chemistry", "Biology",
    "English", "Japanese", "Korean", "Chinese",
    "Programming", "Web Development", "Data Science",
    "Business", "Economics", "Accounting",
    "Music", "Art", "Photography",
]

class FirestoreSeeder:
    def __init__(self, num_users=1000, num_tutors=200):
        self.db = initialize_firestore()
        self.num_users = num_users
        self.num_tutors = num_tutors
        self.user_ids = []
        self.tutor_ids = []
        self.service_ids = []

    def generate_interests(self) -> List[str]:
        num_interests = random.randint(2, 5)
        return random.sample(SUBJECTS, num_interests)

    def generate_specializations(self, subject: str) -> List[str]:
        specializations = {
            "Mathematics": ["Calculus", "Algebra", "Geometry", "Statistics"],
            "Programming": ["Python", "Java", "JavaScript", "C++"],
            "Japanese": ["JLPT N1", "JLPT N2", "Conversation", "Business Japanese"],
        }
        default_specs = ["Beginner", "Intermediate", "Advanced"]
        subject_specs = specializations.get(subject, default_specs)
        num_specs = random.randint(1, min(3, len(subject_specs)))
        return random.sample(subject_specs, num_specs)

    def seed_users(self):
        print("Seeding users...")
        batch = self.db.batch()
        users_ref = self.db.collection('users')

        for _ in range(self.num_users):
            user_id = str(uuid.uuid4())
            self.user_ids.append(user_id)

            user_data = {
                'email': fake.email(),
                'phone_num': fake.phone_number()[:13],
                'city': fake.city(),
                'interests': self.generate_interests(),
                'learning_style': random.choice(LEARNING_STYLES),
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'last_seen': datetime.now()
            }

            batch.set(users_ref.document(user_id), user_data)

        batch.commit()

    def seed_tutors(self):
        print("Seeding tutors...")
        tutors_ref = self.db.collection('tutors')
        services_ref = self.db.collection('tutor_services')

        for _ in range(self.num_tutors):
            tutor_id = str(uuid.uuid4())
            self.tutor_ids.append(tutor_id)

            # Create tutor
            tutor_data = {
                'email': fake.email(),
                'phone_num': fake.phone_number()[:13],
                'latitude': float(fake.latitude()),
                'longitude': float(fake.longitude()),
                'coverage_range': random.randint(1, 20),
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'last_seen': datetime.now()
            }

            tutors_ref.document(tutor_id).set(tutor_data)

            # Create 1-3 services per tutor
            num_services = random.randint(1, 3)
            for _ in range(num_services):
                service_id = str(uuid.uuid4())
                self.service_ids.append(service_id)
                subject = random.choice(SUBJECTS)

                service_data = {
                    'tutor_id': tutor_id,
                    'year_of_experience': random.randint(1, 15),
                    'teaching_style': random.choice(LEARNING_STYLES),
                    'hourly_rate': random.randint(20, 100),
                    'subject': subject,
                    'specialization': self.generate_specializations(subject),
                    'description': fake.paragraph(),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }

                services_ref.document(service_id).set(service_data)

    def seed_ratings(self):
        print("Seeding ratings...")
        ratings_ref = self.db.collection('session_ratings')

        for user_id in self.user_ids:
            num_ratings = random.randint(0, 5)
            for _ in range(num_ratings):
                service_id = random.choice(self.service_ids)

                rating_data = {
                    'user_id': user_id,
                    'service_id': service_id,
                    'message': fake.paragraph(),
                    'rating': random.randint(1, 5),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }

                ratings_ref.document(str(uuid.uuid4())).set(rating_data)

    def run(self):
        try:
            self.seed_users()
            self.seed_tutors()
            self.seed_ratings()

            # Print statistics
            users = len(list(self.db.collection('users').stream()))
            tutors = len(list(self.db.collection('tutors').stream()))
            services = len(list(self.db.collection('tutor_services').stream()))
            ratings = len(list(self.db.collection('session_ratings').stream()))

            print(f"\nSeeding completed:")
            print(f"Users: {users}")
            print(f"Tutors: {tutors}")
            print(f"Services: {services}")
            print(f"Ratings: {ratings}")

        except Exception as e:
            print(f"Error seeding data: {str(e)}")
            raise

if __name__ == "__main__":
    seeder = FirestoreSeeder(num_users=2000, num_tutors=100)
    seeder.run()

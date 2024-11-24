import asyncio
import asyncpg
from faker import Faker
import uuid
from datetime import datetime, timedelta
import random
from typing import List, Dict
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import settings

fake = Faker()

# Constants for data generation
LEARNING_STYLES = ["visual", "auditory", "kinesthetic"]
GENDERS = ["male", "female", "prefer not to say"]
LESSON_TYPES = ["online", "offline", "both"]
ORDER_STATUSES = ["pending", "declined", "scheduled", "completed"]
MESSAGE_TYPES = ["text", "image"]
USER_ROLES = ["learner", "tutor"]

# Sample cities and districts for Japan
CITIES = ["Tokyo", "Osaka", "Yokohama", "Nagoya", "Sapporo", "Fukuoka"]
DISTRICTS = {
    "Tokyo": ["Shibuya", "Shinjuku", "Minato", "Setagaya", "Chiyoda"],
    "Osaka": ["Kita", "Chuo", "Tennoji", "Yodogawa", "Sumiyoshi"],
    "Yokohama": ["Naka", "Tsurumi", "Kanagawa", "Hodogaya", "Isogo"],
    "Nagoya": ["Chikusa", "Higashi", "Kita", "Naka", "Showa"],
    "Sapporo": ["Chuo", "Kita", "Higashi", "Shiroishi", "Toyohira"],
    "Fukuoka": ["Hakata", "Chuo", "Higashi", "Jonan", "Sawara"]
}

SUBJECTS = [
    "Mathematics", "Physics", "Chemistry", "Biology",
    "English", "Japanese", "Korean", "Chinese",
    "Programming", "Web Development", "Data Science",
    "Business", "Economics", "Accounting",
    "Music", "Art", "Photography",
]

class DataSeeder:
    def __init__(self, num_learners=1000, num_tutors=200):
        self.num_learners = num_learners
        self.num_tutors = num_tutors
        self.learner_ids = []
        self.tutor_ids = []
        self.subject_ids = {}
        self.tutory_ids = []
        self.chat_room_ids = []

    def get_random_location(self):
        city = random.choice(CITIES)
        district = random.choice(DISTRICTS[city])
        return city, district

    def generate_availability(self) -> str:
        availability = {}
        days = range(7)
        hours = [f"{h:02d}:00" for h in range(8, 22)]

        for day in days:
            if random.random() > 0.2:
                num_slots = random.randint(1, 5)
                availability[str(day)] = random.sample(hours, num_slots)

        return json.dumps(availability)

    async def seed_subjects(self, conn):
        for subject in tqdm(SUBJECTS, desc="Seeding subjects"):
            subject_id = uuid.uuid4()
            self.subject_ids[subject] = subject_id

            await conn.execute("""
                INSERT INTO subjects (id, name, icon_url, created_at)
                VALUES ($1, $2, $3, $4)
            """,
                subject_id,
                subject,
                f"/icons/subjects/{subject.lower().replace(' ', '-')}.png",
                datetime.now()
            )

    async def seed_learners(self, conn):
        used_emails = set()

        for _ in tqdm(range(self.num_learners), desc="Seeding learners"):
            learner_id = uuid.uuid4()
            self.learner_ids.append(learner_id)

            while True:
                email = f"learner_{uuid.uuid4().hex[:8]}@example.com"
                if email not in used_emails:
                    used_emails.add(email)
                    break

            city, district = self.get_random_location()

            await conn.execute("""
                INSERT INTO learners (
                    id, name, email, password, learning_style, gender,
                    phone_number, city, district, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
            """,
                learner_id,
                fake.name(),
                email,
                "hashed_password_here",
                random.choice(LEARNING_STYLES),
                random.choice(GENDERS),
                fake.phone_number()[:20],
                city,
                district,
                datetime.now()
            )

    async def seed_tutors(self, conn):
        used_emails = set()

        for _ in tqdm(range(self.num_tutors), desc="Seeding tutors"):
            tutor_id = uuid.uuid4()
            self.tutor_ids.append(tutor_id)

            while True:
                email = f"tutor_{uuid.uuid4().hex[:8]}@example.com"
                if email not in used_emails:
                    used_emails.add(email)
                    break

            city, district = self.get_random_location()

            await conn.execute("""
                INSERT INTO tutors (
                    id, name, email, password, gender, phone_number,
                    city, district, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
            """,
                tutor_id,
                fake.name(),
                email,
                "hashed_password_here",
                random.choice(GENDERS),
                fake.phone_number()[:20],
                city,
                district,
                datetime.now()
            )

            # Create tutories
            num_tutories = random.randint(1, 3)
            for _ in range(num_tutories):
                tutory_id = uuid.uuid4()
                self.tutory_ids.append(tutory_id)
                subject = random.choice(SUBJECTS)
                subject_id = self.subject_ids[subject]

                await conn.execute("""
                    INSERT INTO tutories (
                        id, tutor_id, subject_id, about_you, teaching_methodology,
                        hourly_rate, type_lesson, availability, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                """,
                    tutory_id,
                    tutor_id,
                    subject_id,
                    fake.paragraph(),
                    fake.paragraph(),
                    random.randint(20, 100),
                    random.choice(LESSON_TYPES),
                    self.generate_availability(),
                    datetime.now()
                )

    async def seed_orders(self, conn):
        total_orders = len(self.learner_ids) * 2  # Average 2 orders per learner

        with tqdm(total=total_orders, desc="Seeding orders") as pbar:
            for learner_id in self.learner_ids:
                # Generate 0-5 orders per learner
                num_orders = random.randint(0, 5)
                for _ in range(num_orders):
                    tutory_id = random.choice(self.tutory_ids)
                    tutor_id = await conn.fetchval(
                        "SELECT tutor_id FROM tutories WHERE id = $1",
                        tutory_id
                    )

                    await conn.execute("""
                        INSERT INTO orders (
                            id, learner_id, tutor_id, tutory_id, session_time,
                            total_hours, notes, status, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        uuid.uuid4(),
                        learner_id,
                        tutor_id,
                        tutory_id,
                        fake.date_time_between(start_date="-3m", end_date="+1m"),
                        random.randint(1, 4),
                        fake.paragraph() if random.random() > 0.5 else None,
                        random.choice(ORDER_STATUSES),
                        datetime.now()
                    )
                    pbar.update(1)

    async def run(self):
        conn = await asyncpg.connect(settings.POSTGRES_URL)

        try:
            await self.seed_subjects(conn)
            await self.seed_learners(conn)
            await self.seed_tutors(conn)
            await self.seed_orders(conn)

            # Print statistics
            print("\nSeeding completed:")
            print(f"Subjects: {await conn.fetchval('SELECT COUNT(*) FROM subjects')}")
            print(f"Learners: {await conn.fetchval('SELECT COUNT(*) FROM learners')}")
            print(f"Tutors: {await conn.fetchval('SELECT COUNT(*) FROM tutors')}")
            print(f"Tutories: {await conn.fetchval('SELECT COUNT(*) FROM tutories')}")
            print(f"Orders: {await conn.fetchval('SELECT COUNT(*) FROM orders')}")

        finally:
            await conn.close()

if __name__ == "__main__":
    seeder = DataSeeder(num_learners=10000, num_tutors=2000)
    asyncio.run(seeder.run())

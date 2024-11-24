import asyncio
import asyncpg
from faker import Faker
import uuid
from datetime import datetime
import random
from typing import List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import settings

fake = Faker()

# Constants for data generation
LEARNING_STYLES = ["structured", "flexible", "project-based"]
SUBJECTS = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "English",
    "Japanese",
    "Korean",
    "Chinese",
    "Programming",
    "Web Development",
    "Data Science",
    "Business",
    "Economics",
    "Accounting",
    "Music",
    "Art",
    "Photography",
]


class DataSeeder:
    def __init__(self, num_users=1000, num_tutors=200):
        self.num_users = num_users
        self.num_tutors = num_tutors
        self.user_ids = []
        self.tutor_ids = []
        self.service_ids = []

    def generate_interests(self) -> List[str]:
        num_interests = random.randint(2, 5)
        return random.sample(SUBJECTS, num_interests)

    def generate_specializations(self, subject: str) -> List[str]:
        # Generate subject-specific specializations
        specializations = {
            "Mathematics": ["Calculus", "Algebra", "Geometry", "Statistics"],
            "Programming": ["Python", "Java", "JavaScript", "C++"],
            "Japanese": ["JLPT N1", "JLPT N2", "Conversation", "Business Japanese"],
            # Add more subject-specific specializations as needed
        }
        default_specs = ["Beginner", "Intermediate", "Advanced"]

        subject_specs = specializations.get(subject, default_specs)
        num_specs = random.randint(1, min(3, len(subject_specs)))
        return random.sample(subject_specs, num_specs)

    async def seed_users(self, conn):
        print("Seeding users...")
        for _ in range(self.num_users):
            user_id = uuid.uuid4()
            self.user_ids.append(user_id)

            await conn.execute(
                """
                INSERT INTO users (
                    id, email, phone_num, city, interest, learning_style,
                    created_at, updated_at, last_seen
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7, $7)
            """,
                user_id,
                fake.email(),
                fake.phone_number()[:13],
                fake.city(),
                self.generate_interests(),
                random.choice(LEARNING_STYLES),
                datetime.now(),
            )

    async def seed_tutors(self, conn):
        print("Seeding tutors...")
        for _ in range(self.num_tutors):
            tutor_id = uuid.uuid4()
            self.tutor_ids.append(tutor_id)

            # Create tutor
            await conn.execute(
                """
                INSERT INTO tutor (
                    id, email, phone_num, latitude, longitude, coverage_range,
                    created_at, updated_at, last_seen
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7, $7)
                """,
                tutor_id,
                fake.email(),
                fake.phone_number()[:13],
                float(fake.latitude()),  # Store latitude directly
                float(fake.longitude()),  # Store longitude directly
                random.randint(1, 20),
                datetime.now(),
            )

            # Create 1-3 services per tutor
            num_services = random.randint(1, 3)
            for _ in range(num_services):
                service_id = uuid.uuid4()
                self.service_ids.append(service_id)
                subject = random.choice(SUBJECTS)

                await conn.execute(
                    """
                    INSERT INTO tutor_service (
                        id, tutor_id, year_of_experience, teaching_style,
                        hourly_rate, subject, specialization, description,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                """,
                    service_id,
                    tutor_id,
                    random.randint(1, 15),
                    random.choice(LEARNING_STYLES),
                    random.randint(20, 100),  # hourly rate
                    subject,
                    self.generate_specializations(subject),
                    fake.paragraph(),
                    datetime.now(),
                )

    async def seed_orders_and_ratings(self, conn):
        print("Seeding orders and ratings...")
        for user_id in self.user_ids:
            # Generate 0-5 orders per user
            num_orders = random.randint(0, 5)
            for _ in range(num_orders):
                service_id = random.choice(self.service_ids)
                tutor_id = await conn.fetchval(
                    "SELECT tutor_id FROM tutor_service WHERE id = $1", service_id
                )

                # Create order
                session_time = fake.date_time_between(start_date="-3m", end_date="+1m")
                status = random.choice(["completed", "scheduled", "pending"])

                await conn.execute(
                    """
                    INSERT INTO "order" (
                        user_id, tutor_id, service_id, session_time,
                        total_hour, status, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                """,
                    user_id,
                    tutor_id,
                    service_id,
                    session_time,
                    random.randint(1, 4),
                    status,
                    datetime.now(),
                )

                # Add rating for completed orders
                if status == "completed":
                    await conn.execute(
                        """
                        INSERT INTO session_rating (
                            session_id, user_id, service_id, message,
                            session_rating, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $6)
                    """,
                        uuid.uuid4(),  # session_id
                        user_id,
                        service_id,
                        fake.paragraph(),
                        random.randint(1, 5),
                        datetime.now(),
                    )

    async def run(self):
        conn = await asyncpg.connect(settings.POSTGRES_URL)

        try:
            # Seed data
            await self.seed_users(conn)
            await self.seed_tutors(conn)
            await self.seed_orders_and_ratings(conn)

            # Print statistics (update this query to use 'users')
            users = await conn.fetchval("SELECT COUNT(*) FROM users")
            tutors = await conn.fetchval("SELECT COUNT(*) FROM tutor")
            services = await conn.fetchval("SELECT COUNT(*) FROM tutor_service")
            orders = await conn.fetchval('SELECT COUNT(*) FROM "order"')
            ratings = await conn.fetchval("SELECT COUNT(*) FROM session_rating")

            print(f"\nSeeding completed:")
            print(f"Users: {users}")
            print(f"Tutors: {tutors}")
            print(f"Services: {services}")
            print(f"Orders: {orders}")
            print(f"Ratings: {ratings}")

        finally:
            await conn.close()


if __name__ == "__main__":
    seeder = DataSeeder(num_users=10000, num_tutors=2000)
    asyncio.run(seeder.run())

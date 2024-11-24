import asyncio
import asyncpg
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import settings

async def init_db():
    conn = await asyncpg.connect(settings.POSTGRES_URL)

    try:
        # Drop existing tables and types if they exist
        await conn.execute("""
            DROP TABLE IF EXISTS recommendation_cache CASCADE;
            DROP TABLE IF EXISTS session_rating CASCADE;
            DROP TABLE IF EXISTS orders CASCADE;
            DROP TABLE IF EXISTS tutories CASCADE;
            DROP TABLE IF EXISTS tutors CASCADE;
            DROP TABLE IF EXISTS learners CASCADE;
            DROP TABLE IF EXISTS subjects CASCADE;
            DROP TYPE IF EXISTS learning_style_enum CASCADE;
            DROP TYPE IF EXISTS gender_enum CASCADE;
            DROP TYPE IF EXISTS type_lesson_enum CASCADE;
            DROP TYPE IF EXISTS order_status_enum CASCADE;
        """)

        # Create enum types
        await conn.execute("""
            DO $$ BEGIN
                CREATE TYPE gender_enum AS ENUM ('male', 'female', 'prefer not to say');
                CREATE TYPE learning_style_enum AS ENUM ('visual', 'auditory', 'kinesthetic');
                CREATE TYPE type_lesson_enum AS ENUM ('online', 'offline', 'both');
                CREATE TYPE order_status_enum AS ENUM ('pending', 'declined', 'scheduled', 'completed');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)

        # Create SUBJECTS table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                icon_url VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)

        # Create LEARNERS table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS learners (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                learning_style learning_style_enum,
                gender gender_enum,
                phone_number VARCHAR(20),
                city VARCHAR(255),
                district VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP
            )
        """)

        # Create TUTORS table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tutors (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                gender gender_enum,
                phone_number VARCHAR(20),
                city VARCHAR(255),
                district VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP
            )
        """)

        # Create TUTORIES table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tutories (
                id UUID PRIMARY KEY,
                tutor_id UUID REFERENCES tutors(id) ON DELETE CASCADE,
                subject_id UUID REFERENCES subjects(id) ON DELETE CASCADE,
                about_you TEXT NOT NULL,
                teaching_methodology TEXT NOT NULL,
                hourly_rate INTEGER NOT NULL,
                type_lesson type_lesson_enum,
                availability JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP
            )
        """)

        # Create ORDERS table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id UUID PRIMARY KEY,
                learner_id UUID REFERENCES learners(id) ON DELETE CASCADE,
                tutor_id UUID REFERENCES tutors(id) ON DELETE CASCADE,
                tutory_id UUID REFERENCES tutories(id) ON DELETE CASCADE,
                session_time TIMESTAMP NOT NULL,
                total_hours INTEGER NOT NULL,
                notes TEXT,
                status order_status_enum,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)

        # Create recommendation cache table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_cache (
                learner_id UUID PRIMARY KEY REFERENCES learners(id),
                recommendations JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        await conn.execute("""
            -- Indexes for SUBJECTS table
            CREATE INDEX IF NOT EXISTS idx_subjects_name ON subjects(name);

            -- Indexes for LEARNERS table
            CREATE INDEX IF NOT EXISTS idx_learners_learning_style ON learners(learning_style);
            CREATE INDEX IF NOT EXISTS idx_learners_city ON learners(city);
            CREATE INDEX IF NOT EXISTS idx_learners_district ON learners(district);

            -- Indexes for TUTORS table
            CREATE INDEX IF NOT EXISTS idx_tutors_city ON tutors(city);
            CREATE INDEX IF NOT EXISTS idx_tutors_district ON tutors(district);

            -- Indexes for TUTORIES table
            CREATE INDEX IF NOT EXISTS idx_tutories_tutor ON tutories(tutor_id);
            CREATE INDEX IF NOT EXISTS idx_tutories_subject ON tutories(subject_id);
            CREATE INDEX IF NOT EXISTS idx_tutories_hourly_rate ON tutories(hourly_rate);

            -- Indexes for ORDERS table
            CREATE INDEX IF NOT EXISTS idx_orders_learner ON orders(learner_id);
            CREATE INDEX IF NOT EXISTS idx_orders_tutor ON orders(tutor_id);
            CREATE INDEX IF NOT EXISTS idx_orders_tutory ON orders(tutory_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        """)

        print("Database initialization completed successfully!")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())

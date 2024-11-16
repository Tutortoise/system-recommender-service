import asyncio
import asyncpg
import sys
from pathlib import Path

# Add parent directory to path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from config import settings


async def init_db():
    conn = await asyncpg.connect(settings.POSTGRES_URL)

    try:
        # Drop existing tables and types if they exist
        await conn.execute(
            """
            DROP TABLE IF EXISTS recommendation_cache CASCADE;
            DROP TABLE IF EXISTS session_rating CASCADE;
            DROP TABLE IF EXISTS "order" CASCADE;
            DROP TABLE IF EXISTS tutor_service CASCADE;
            DROP TABLE IF EXISTS tutor CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
            DROP TYPE IF EXISTS learning_style_enum CASCADE;
            DROP TYPE IF EXISTS order_status_enum CASCADE;
        """
        )

        # Create enum types
        await conn.execute(
            """
            DO $$ BEGIN
                CREATE TYPE learning_style_enum AS ENUM ('structured', 'flexible', 'project-based');
                CREATE TYPE order_status_enum AS ENUM ('pending', 'scheduled', 'completed');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """
        )

        # Create USERS table first (no dependencies)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR NOT NULL,
                phone_num VARCHAR(13),
                city VARCHAR,
                interest VARCHAR[],
                learning_style learning_style_enum,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_seen TIMESTAMP
            )
        """
        )

        # Create TUTOR table (no dependencies)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tutor (
                id UUID PRIMARY KEY,
                email VARCHAR NOT NULL,
                phone_num VARCHAR(13),
                services UUID[],
                latitude FLOAT,           -- Change from POINT to separate coordinates
                longitude FLOAT,          -- Change from POINT to separate coordinates
                coverage_range SMALLINT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_seen TIMESTAMP
            )
        """
        )

        # Create TUTOR_SERVICE table (depends on tutor)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tutor_service (
                id UUID PRIMARY KEY,
                tutor_id UUID REFERENCES tutor(id),
                year_of_experience SMALLINT,
                teaching_style learning_style_enum,
                hourly_rate BIGINT,
                subject VARCHAR,
                specialization VARCHAR[],
                description TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """
        )

        # Create ORDER table (depends on users, tutor, and tutor_service)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS "order" (
                user_id UUID REFERENCES users(id),
                tutor_id UUID REFERENCES tutor(id),
                service_id UUID REFERENCES tutor_service(id),
                session_time TIMESTAMP,
                total_hour SMALLINT,
                status order_status_enum,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """
        )

        # Create SESSION_RATING table (depends on users and tutor_service)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_rating (
                session_id UUID PRIMARY KEY,
                user_id UUID REFERENCES users(id),
                service_id UUID REFERENCES tutor_service(id),
                message TEXT,
                session_rating SMALLINT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """
        )

        # Create recommendation cache table (depends on users)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_cache (
                user_id UUID PRIMARY KEY REFERENCES users(id),
                recommendations JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes after all tables are created
        await conn.execute(
            """
            -- Indexes for USERS table
            CREATE INDEX IF NOT EXISTS idx_users_interest ON users USING gin(interest);
            CREATE INDEX IF NOT EXISTS idx_users_learning_style ON users(learning_style);
            CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen);

            -- Indexes for TUTOR table
            CREATE INDEX IF NOT EXISTS idx_tutor_services ON tutor USING gin(services);
            CREATE INDEX IF NOT EXISTS idx_tutor_latitude ON tutor(latitude);      -- New index for latitude
            CREATE INDEX IF NOT EXISTS idx_tutor_longitude ON tutor(longitude);    -- New index for longitude
            CREATE INDEX IF NOT EXISTS idx_tutor_last_seen ON tutor(last_seen);

            -- Indexes for TUTOR_SERVICE table
            CREATE INDEX IF NOT EXISTS idx_tutor_service_tutor ON tutor_service(tutor_id);
            CREATE INDEX IF NOT EXISTS idx_tutor_service_subject ON tutor_service(subject);
            CREATE INDEX IF NOT EXISTS idx_tutor_service_specialization ON tutor_service USING gin(specialization);

            -- Indexes for ORDER table
            CREATE INDEX IF NOT EXISTS idx_order_user ON "order"(user_id);
            CREATE INDEX IF NOT EXISTS idx_order_tutor ON "order"(tutor_id);
            CREATE INDEX IF NOT EXISTS idx_order_service ON "order"(service_id);
            CREATE INDEX IF NOT EXISTS idx_order_status ON "order"(status);

            -- Indexes for SESSION_RATING table
            CREATE INDEX IF NOT EXISTS idx_session_rating_user ON session_rating(user_id);
            CREATE INDEX IF NOT EXISTS idx_session_rating_service ON session_rating(service_id);
            CREATE INDEX IF NOT EXISTS idx_session_rating ON session_rating(session_rating);

            -- Indexes for recommendation_cache table
            CREATE INDEX IF NOT EXISTS idx_tutor_service_tutor_id ON tutor_service(tutor_id);
            CREATE INDEX IF NOT EXISTS idx_session_rating_service_id ON session_rating(service_id);
            CREATE INDEX IF NOT EXISTS idx_tutor_service_subject ON tutor_service(subject);
            CREATE INDEX IF NOT EXISTS idx_tutor_service_composite
                ON tutor_service(tutor_id, subject, teaching_style);
            """
        )

        print("Database initialization completed successfully!")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_db())

import asyncio
import asyncpg
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import settings


async def init_db():
    conn = await asyncpg.connect(settings.POSTGRES_URL)

    try:
        # Drop existing tables and types
        await conn.execute("""
            DROP TABLE IF EXISTS reviews CASCADE;
            DROP TABLE IF EXISTS chat_messages CASCADE;
            DROP TABLE IF EXISTS chat_rooms CASCADE;
            DROP TABLE IF EXISTS fcm_tokens CASCADE;
            DROP TABLE IF EXISTS orders CASCADE;
            DROP TABLE IF EXISTS interests CASCADE;
            DROP TABLE IF EXISTS tutories CASCADE;
            DROP TABLE IF EXISTS tutors CASCADE;
            DROP TABLE IF EXISTS learners CASCADE;
            DROP TABLE IF EXISTS categories CASCADE;

            DROP TYPE IF EXISTS message_type_enum CASCADE;
            DROP TYPE IF EXISTS user_role_enum CASCADE;
            DROP TYPE IF EXISTS learning_style_enum CASCADE;
            DROP TYPE IF EXISTS gender_enum CASCADE;
            DROP TYPE IF EXISTS type_lesson_enum CASCADE;
            DROP TYPE IF EXISTS status_enum CASCADE;
        """)

        # Create enum types
        await conn.execute("""
            DO $$ BEGIN
                CREATE TYPE gender_enum AS ENUM ('male', 'female', 'prefer not to say');
                CREATE TYPE learning_style_enum AS ENUM ('visual', 'auditory', 'kinesthetic');
                CREATE TYPE type_lesson_enum AS ENUM ('online', 'offline', 'both');
                CREATE TYPE status_enum AS ENUM ('pending', 'declined', 'scheduled', 'completed');
                CREATE TYPE message_type_enum AS ENUM ('text', 'image');
                CREATE TYPE user_role_enum AS ENUM ('learner', 'tutor');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)

        # Create tables
        await conn.execute("""
            -- categories table
            CREATE TABLE IF NOT EXISTS categories (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                icon_url VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );

            -- Learners table
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
            );

            -- Interests table (many-to-many between learners and categories)
            CREATE TABLE IF NOT EXISTS interests (
                learner_id UUID REFERENCES learners(id) ON DELETE CASCADE,
                category_id UUID REFERENCES categories(id) ON DELETE CASCADE,
                PRIMARY KEY (learner_id, category_id)
            );

            -- Tutors table
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
            );

            -- Tutories table
            CREATE TABLE IF NOT EXISTS tutories (
                id UUID PRIMARY KEY,
                tutor_id UUID REFERENCES tutors(id) ON DELETE CASCADE,
                category_id UUID REFERENCES categories(id) ON DELETE CASCADE,
                name VARCHAR(50) NOT NULL,
                about_you TEXT NOT NULL,
                teaching_methodology TEXT NOT NULL,
                hourly_rate INTEGER NOT NULL,
                type_lesson type_lesson_enum NOT NULL,
                is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP
            );

            -- Orders table
            CREATE TABLE IF NOT EXISTS orders (
                id UUID PRIMARY KEY,
                learner_id UUID REFERENCES learners(id) ON DELETE CASCADE,
                tutories_id UUID REFERENCES tutories(id) ON DELETE CASCADE,
                session_time TIMESTAMP NOT NULL,
                estimated_end_time TIMESTAMP,
                total_hours INTEGER NOT NULL,
                notes TEXT,
                type_lesson type_lesson_enum NOT NULL,
                status status_enum,
                price INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );

            -- Chat rooms table
            CREATE TABLE IF NOT EXISTS chat_rooms (
                id UUID PRIMARY KEY,
                learner_id UUID REFERENCES learners(id),
                tutor_id UUID REFERENCES tutors(id),
                last_message_at TIMESTAMP NOT NULL DEFAULT NOW(),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );

            -- Chat messages table
            CREATE TABLE IF NOT EXISTS chat_messages (
                id UUID PRIMARY KEY,
                room_id UUID REFERENCES chat_rooms(id),
                sender_id UUID NOT NULL,
                sender_role user_role_enum NOT NULL,
                content TEXT NOT NULL,
                type message_type_enum NOT NULL,
                sent_at TIMESTAMP NOT NULL DEFAULT NOW(),
                is_read BOOLEAN NOT NULL DEFAULT FALSE
            );

            -- FCM tokens table
            CREATE TABLE IF NOT EXISTS fcm_tokens (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP,
                UNIQUE(user_id, token)
            );

            -- Reviews table
            CREATE TABLE IF NOT EXISTS reviews (
                id UUID PRIMARY KEY,
                order_id UUID UNIQUE REFERENCES orders(id) ON DELETE CASCADE,
                rating INTEGER NOT NULL,
                message TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """)

        # Create indexes
        await conn.execute("""
            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);
            CREATE INDEX IF NOT EXISTS idx_learners_city_district ON learners(city, district);
            CREATE INDEX IF NOT EXISTS idx_tutors_city_district ON tutors(city, district);
            CREATE INDEX IF NOT EXISTS idx_tutories_tutor_category ON tutories(tutor_id, category_id);
            CREATE INDEX IF NOT EXISTS idx_tutories_hourly_rate ON tutories(hourly_rate);
            CREATE INDEX IF NOT EXISTS idx_tutories_type_lesson ON tutories(type_lesson);
            CREATE INDEX IF NOT EXISTS idx_tutories_created_at ON tutories(created_at);
            CREATE INDEX IF NOT EXISTS idx_orders_learner ON orders(learner_id);
            CREATE INDEX IF NOT EXISTS idx_orders_tutories ON orders(tutories_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_session_time ON orders(session_time);
            CREATE INDEX IF NOT EXISTS idx_chat_rooms_participants ON chat_rooms(learner_id, tutor_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_room ON chat_messages(room_id, sent_at);
            CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
            CREATE INDEX IF NOT EXISTS idx_fcm_tokens_user ON fcm_tokens(user_id);
        """)

        print("Database initialization completed successfully!")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_db())

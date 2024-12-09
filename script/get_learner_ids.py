import asyncio
import asyncpg
import sys
from pathlib import Path
import random
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from config import settings


async def get_random_learner_ids(
    limit: int = 10, min_interests: int = 1, has_orders: bool = False
) -> List[str]:
    """
    Get random learner IDs from database with optional filters

    Args:
        limit: Number of learner IDs to return
        min_interests: Minimum number of interests a learner should have
        has_orders: If True, only return learners who have placed orders

    Returns:
        List of learner IDs as strings
    """
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)

        # Build query based on filters
        query = """
            WITH learner_stats AS (
                SELECT
                    l.id,
                    COUNT(DISTINCT i.category_id) as interest_count,
                    COUNT(DISTINCT o.id) as order_count
                FROM learners l
                LEFT JOIN interests i ON l.id = i.learner_id
                LEFT JOIN orders o ON l.id = o.learner_id
                GROUP BY l.id
                HAVING
                    COUNT(DISTINCT i.category_id) >= $1
                    {order_filter}
            )
            SELECT id::text
            FROM learner_stats
            ORDER BY random()
            LIMIT $2
        """

        # Add order filter if requested
        order_filter = "AND COUNT(DISTINCT o.id) > 0" if has_orders else ""
        query = query.format(order_filter=order_filter)

        # Execute query
        rows = await conn.fetch(query, min_interests, limit)
        learner_ids = [row["id"] for row in rows]

        # Print summary
        print(f"\nFetched {len(learner_ids)} learner IDs with filters:")
        print(f"- Minimum interests: {min_interests}")
        print(f"- Has orders: {has_orders}")

        if learner_ids:
            print("\nSample IDs:")
            for i, lid in enumerate(learner_ids[:5], 1):
                print(f"{i}. {lid}")

            if len(learner_ids) > 5:
                print(f"... and {len(learner_ids) - 5} more")
        else:
            print("\nNo learners found matching criteria!")

        return learner_ids

    except Exception as e:
        print(f"Error getting learner IDs: {str(e)}")
        raise
    finally:
        await conn.close()


async def get_learner_details(learner_id: str) -> Optional[dict]:
    """Get detailed information about a specific learner"""
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)

        # Get learner details with related data
        learner = await conn.fetchrow(
            """
            SELECT
                l.*,
                array_agg(DISTINCT c.name) as interests,
                COUNT(DISTINCT o.id) as total_orders,
                COUNT(DISTINCT CASE WHEN o.status = 'completed' THEN o.id END) as completed_orders
            FROM learners l
            LEFT JOIN interests i ON l.id = i.learner_id
            LEFT JOIN categories c ON i.category_id = c.id
            LEFT JOIN orders o ON l.id = o.learner_id
            WHERE l.id = $1
            GROUP BY l.id
        """,
            learner_id,
        )

        if learner:
            return dict(learner)
        return None

    except Exception as e:
        print(f"Error getting learner details: {str(e)}")
        raise
    finally:
        await conn.close()


async def main():
    # Get random learners with different criteria
    print("\n=== Getting Random Learner IDs ===")

    # Basic random selection
    basic_ids = await get_random_learner_ids(limit=5)

    # Learners with at least 3 interests
    interested_ids = await get_random_learner_ids(limit=5, min_interests=3)

    # Active learners (with orders and interests)
    active_ids = await get_random_learner_ids(limit=5, min_interests=2, has_orders=True)

    # Get details for one random learner
    if active_ids:
        print("\n=== Sample Learner Details ===")
        sample_id = random.choice(active_ids)
        details = await get_learner_details(sample_id)

        if details:
            print(f"\nLearner ID: {details['id']}")
            print(f"Name: {details['name']}")
            print(f"Email: {details['email']}")
            print(f"Learning Style: {details['learning_style']}")
            print(f"Location: {details['city']}, {details['district']}")
            print(f"Interests: {', '.join(details['interests'])}")
            print(f"Total Orders: {details['total_orders']}")
            print(f"Completed Orders: {details['completed_orders']}")


if __name__ == "__main__":
    asyncio.run(main())

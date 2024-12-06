from collections import defaultdict
from typing import Dict, List
import time


class InteractionTracker:
    def __init__(self):
        self.interactions = defaultdict(list)
        self.max_interactions = 1000
        self.interaction_weight = 0.3
        self.category_interactions = defaultdict(lambda: defaultdict(int))

    def add_interaction(
        self,
        user_id: str,
        tutory_id: str,
        tutor_id: str,
        category_id: str,
        category_name: str,
    ):
        """Add new click interaction"""
        timestamp = time.time()

        interaction = {
            "tutory_id": tutory_id,
            "tutor_id": tutor_id,
            "category_id": category_id,
            "category_name": category_name,
            "timestamp": timestamp,
            "weight": self.interaction_weight,
        }

        self.interactions[user_id].append(interaction)
        self.category_interactions[user_id][category_id] += self.interaction_weight

        if len(self.interactions[user_id]) > self.max_interactions:
            self.interactions[user_id] = self.interactions[user_id][
                -self.max_interactions :
            ]

        return self.interaction_weight

    def get_category_preferences(self, user_id: str) -> Dict[str, float]:
        """Get user's category preferences based on interactions"""
        preferences = self.category_interactions.get(user_id, {})
        if not preferences:
            return {}

        # Normalize preferences
        total = sum(preferences.values())
        return {cat_id: count / total for cat_id, count in preferences.items()}

    def get_user_interactions(self, user_id: str) -> List[Dict]:
        """Get user's interaction history"""
        return self.interactions.get(user_id, [])

    def clear_old_interactions(self, max_age_hours: int = 24):
        """Clear interactions older than max_age_hours"""
        current_time = time.time()
        for user_id in list(self.interactions.keys()):
            current_interactions = [
                interaction
                for interaction in self.interactions[user_id]
                if (current_time - interaction["timestamp"]) < (max_age_hours * 3600)
            ]

            # Update category interactions if interactions were removed
            if len(current_interactions) < len(self.interactions[user_id]):
                self.category_interactions[user_id].clear()
                for interaction in current_interactions:
                    self.category_interactions[user_id][
                        interaction["category_id"]
                    ] += interaction["weight"]

            self.interactions[user_id] = current_interactions

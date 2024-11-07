import json
from uuid import UUID
from datetime import datetime


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def serialize_for_json(data):
    """Convert a dictionary with UUID values to strings"""
    if isinstance(data, list):
        return [serialize_for_json(item) for item in data]
    if isinstance(data, dict):
        return {
            key: (
                str(value)
                if isinstance(value, UUID)
                else (
                    value.isoformat()
                    if isinstance(value, datetime)
                    else (
                        serialize_for_json(value)
                        if isinstance(value, (dict, list))
                        else value
                    )
                )
            )
            for key, value in data.items()
        }
    return data

from datetime import datetime, timezone
import math

DECAY_HALF_LIFE_DAYS = 30

def days_since(timestamp: str) -> float:
    past = datetime.fromisoformat(timestamp)

    # Force UTC if timestamp is naive
    if past.tzinfo is None:
        past = past.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return (now - past).total_seconds() / 86400


def decay(confidence: float, days: float) -> float:
    return confidence * math.exp(-days / DECAY_HALF_LIFE_DAYS)

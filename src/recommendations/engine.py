# src/recommendations/engine.py
# Lightweight local recommendation engine — no ML required.
# Scores courses against a user profile using keyword overlap + level matching.

import logging
from typing import Optional
from .courses import COURSE_CATALOG

logger = logging.getLogger(__name__)

# Level hierarchy for scoring distance
LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


def recommend_courses(
    goal: Optional[str],
    level: Optional[str],
    career: Optional[str],
    top_n: int = 3,
) -> list[dict]:
    """
    Score every course in the catalog against the user profile and return
    the top_n best matches with a plain-English reason.

    Scoring (max 100):
    - Topic keyword overlap with goal/career  → up to 60 pts
    - Level match                             → up to 30 pts
    - Course rating                           → up to 10 pts
    """
    goal_tokens = _tokenize(goal)
    career_tokens = _tokenize(career)
    user_level = (level or "beginner").lower().strip()
    user_level_idx = LEVEL_ORDER.get(user_level, 0)

    scored = []
    for course in COURSE_CATALOG:
        score = 0
        reasons = []

        # ── Topic relevance (60 pts) ─────────────────────────────────────
        topic_tokens = set(course["topics"])
        goal_hits = topic_tokens & goal_tokens
        career_hits = topic_tokens & career_tokens

        if goal_hits:
            pts = min(40, len(goal_hits) * 15)
            score += pts
            reasons.append(f"covers {', '.join(goal_hits)}")
        if career_hits:
            pts = min(20, len(career_hits) * 10)
            score += pts
            reasons.append(f"relevant for {career or 'your career'}")

        # ── Level match (30 pts) ─────────────────────────────────────────
        course_level_idx = LEVEL_ORDER.get(course["level"], 0)
        level_diff = abs(user_level_idx - course_level_idx)
        level_pts = max(0, 30 - level_diff * 15)
        score += level_pts
        if level_diff == 0:
            reasons.append(f"perfect {user_level} level")
        elif level_diff == 1:
            reasons.append(f"close to your {user_level} level")

        # ── Rating bonus (10 pts) ────────────────────────────────────────
        rating_pts = round((course["rating"] - 4.0) / 1.0 * 10)
        score += max(0, rating_pts)

        if score > 0:
            scored.append((score, course, reasons))

    # Sort by score descending, then by rating as tiebreaker
    scored.sort(key=lambda x: (x[0], x[1]["rating"]), reverse=True)

    results = []
    for score, course, reasons in scored[:top_n]:
        reason_str = (
            f"Great match — {'; '.join(reasons)}." if reasons else "Good general fit."
        )
        results.append(
            {
                "id": course["id"],
                "title": course["title"],
                "provider": course["provider"],
                "level": course["level"],
                "duration": course["duration"],
                "rating": course["rating"],
                "description": course["description"],
                "url": course["url"],
                "reason": reason_str,
                "score": score,
            }
        )

    logger.info(
        f"Recommendations for goal='{goal}' level='{level}' career='{career}': "
        f"{[r['title'] for r in results]}"
    )
    return results


def _tokenize(text: Optional[str]) -> set:
    """Lowercase and split text into individual word tokens."""
    if not text:
        return set()
    return set(text.lower().replace("-", " ").split())

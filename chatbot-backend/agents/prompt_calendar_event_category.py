"""Prompt builder for LLM-based calendar event category detection."""

import json
from typing import Any, Dict, List


def build_calendar_event_category_detection_prompt(
    events_payload: List[Dict[str, Any]],
    allowed_categories: List[str],
) -> str:
    """
    Build a prompt that asks the model to assign one category per event index.

    Args:
        events_payload: Each item should include at least: index (int), summary (str),
            and optionally description, location, event_type.
        allowed_categories: Sorted or ordered list of allowed UPPERCASE category codes.
    """
    allowed_json = json.dumps(allowed_categories, ensure_ascii=False)
    events_json = json.dumps(events_payload, ensure_ascii=False)
    return f"""You classify calendar events into exactly one category code per event.

Allowed category codes (use ONLY these strings, UPPERCASE as given):
{allowed_json}

Input events (JSON array). Each object has:
- index: integer position in the batch (0-based)
- summary: event title (required for inference)
- description: optional details
- location: optional
- event_type: optional

Input:
{events_json}

Rules:
1. For every index present in Input, output exactly one category from the allowed list.
2. Pick the best semantic match from summary + description + location + event_type.
3. Physical activities (e.g. gym, workout, running, jogging, cycling, swimming, yoga, sports, hiking, fitness training) belong to HEALTH.
4. If the event is ambiguous or does not fit any specific label confidently, use OTHER.
5. Do not invent codes outside the allowed list.
6. Do not add commentary outside JSON.

Output format (JSON only):
{{
  "categories": [
    {{ "index": 0, "category": "MEETINGS" }},
    {{ "index": 1, "category": "OTHER" }}
  ]
}}

Respond with JSON only, no markdown fences."""

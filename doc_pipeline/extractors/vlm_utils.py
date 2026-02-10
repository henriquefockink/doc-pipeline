"""Shared utilities for VLM response parsing."""

import json
import re


def parse_vlm_json(text: str) -> dict:
    """Extract and parse JSON from VLM response text.

    Handles:
    - JSON wrapped in ```json ... ``` code blocks
    - Raw JSON objects in free-form text
    - Returns empty dict on parse failure
    """
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1)

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

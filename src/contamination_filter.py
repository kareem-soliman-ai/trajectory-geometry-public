"""Utilities for truncating generations and extracting clean numeric answers.

The research programme uses aggressive response cleaning to avoid scoring or
analysing post-answer hallucinations, prompt restarts, and verbose spillover.
These helpers are intentionally lightweight so they can be reused in notebooks,
batch inference scripts, and downstream audit code.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional


DIRECT_REASONING_MARKERS = (
    "step",
    "first",
    "multiply",
    "calculate",
    "because",
    "therefore",
)


def truncate_direct_response(text: str) -> str:
    """Return the first plausible numeric direct-answer substring."""

    text = text.strip()
    match = re.search(r"(?:Answer|Result|is|:)\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(-?\d+)", text)
    if match:
        return match.group(1)
    return text


def truncate_cot_response(text: str) -> str:
    """Trim CoT output at the first final-answer boundary."""

    match = re.search(r"ANSWER:\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return text[: match.end()]

    match = re.search(
        r"(?:the|final)\s*(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return text[: match.end()]

    for pattern in (r"\n\s*(?:Question|Calculate|Q:)", r"(\d)\1{5,}"):
        match = re.search(pattern, text)
        if match:
            return text[: match.start()]
    return text


def parse_numeric_answer(text: str, condition: str) -> Optional[int]:
    """Parse a numeric answer after truncation."""

    if condition == "direct":
        clean = truncate_direct_response(text)
        numbers = re.findall(r"-?\d+", clean)
        return int(numbers[0]) if numbers else None

    clean = truncate_cot_response(text)
    match = re.search(r"ANSWER:\s*(-?\d+)", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"-?\d+", clean)
    return int(numbers[-1]) if numbers else None


def find_clean_token_boundary(generated_tokens: Iterable[int], tokenizer, clean_text: str) -> int:
    """Approximate the token boundary matching the cleaned response string."""

    tokens = list(generated_tokens)
    for idx in range(len(tokens), 0, -1):
        decoded = tokenizer.decode(tokens[:idx], skip_special_tokens=True)
        if len(decoded.strip()) <= len(clean_text.strip()):
            return idx
    return len(tokens)


def validate_response(response: str, clean_response: str, condition: str) -> list[str]:
    """Flag common contamination patterns for auditing."""

    issues: list[str] = []
    if len(response) > len(clean_response) * 2 and len(response) > 50:
        issues.append("response_much_longer_than_clean")

    if condition == "direct" and any(marker in response.lower() for marker in DIRECT_REASONING_MARKERS):
        issues.append("direct_contains_reasoning")

    if re.search(r"(\d)\1{5,}", response):
        issues.append("digit_repetition")

    return issues

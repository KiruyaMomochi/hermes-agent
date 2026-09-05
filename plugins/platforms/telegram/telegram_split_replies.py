"""Telegram --- delimiter-based reply splitting."""
import re


def split_reply_delimited(
    text: str,
    *,
    max_parts: int = 8,
) -> list[tuple[str, int]]:
    """Split on standalone --- lines outside code fences.
    
    Returns list of (content, preceding_dash_count) tuples. The dash count
    controls pause duration between bubbles: --- is the base delay, ----
    is longer, etc.
    """
    if not text or max_parts <= 1:
        return [(text, 0)]

    parts: list[tuple[str, int]] = []
    current: list[str] = []
    preceding_dash_count = 0
    in_code_fence = False
    saw_separator = False

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
        separator = re.fullmatch(r"-{3,}", stripped)
        if separator and not in_code_fence:
            saw_separator = True
            part = "".join(current).strip()
            if part:
                parts.append((part, preceding_dash_count))
                preceding_dash_count = len(stripped)
            current = []
            continue
        current.append(line)

    if not saw_separator:
        return [(text, 0)]
    tail = "".join(current).strip()
    if tail:
        parts.append((tail, preceding_dash_count))
    if not parts:
        return [(text, 0)]
    if len(parts) > max_parts:
        head = parts[: max_parts - 1]
        overflow = parts[max_parts - 1 :]
        merged = "\n\n".join(part for part, _ in overflow)
        return head + [(merged, overflow[0][1])]
    return parts


def split_reply_delay_seconds(base_delay: float, dash_count: int) -> float:
    """Scale pause after --- by dash count. --- is base, ---- is 1.5x, etc."""
    if dash_count < 3:
        return base_delay
    return base_delay * min(5.0, 1.0 + (dash_count - 3) * 0.5)

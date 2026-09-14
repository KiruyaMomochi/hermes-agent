"""Telegram --- delimiter-based reply splitting."""
import re


def split_reply_delimited(
    text: str,
    *,
    max_parts: int = 8,
    with_offsets: bool = False,
):
    """Split on standalone --- lines outside code fences.

    Returns list of (content, preceding_dash_count) tuples. The dash count
    controls pause duration between bubbles: --- is the base delay, ----
    is longer, etc.

    ``with_offsets=True`` appends each part's start offset in the ORIGINAL
    ``text`` as a third tuple element (content, dash_count, start_offset), so a
    partial fanout failure can slice ``text`` into an exact delivered prefix and
    an unsent, re-splittable suffix. Offsets point at the first non-stripped
    character of the part's body (delimiters precede it), so ``text[:offset]``
    of part N is a genuine prefix ending after part N-1's delimiter.
    """
    if not text or max_parts <= 1:
        return [(text, 0, 0)] if with_offsets else [(text, 0)]

    # (content, preceding_dash_count, start_offset)
    parts: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_start = 0  # offset in text where the current segment's lines begin
    preceding_dash_count = 0
    in_code_fence = False
    saw_separator = False
    pos = 0  # running character offset into text

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
        separator = re.fullmatch(r"-{3,}", stripped)
        if separator and not in_code_fence:
            saw_separator = True
            raw = "".join(current)
            part = raw.strip()
            if part:
                # Offset of the stripped body inside the raw segment (skip leading whitespace).
                lead = len(raw) - len(raw.lstrip())
                parts.append((part, preceding_dash_count, current_start + lead))
                preceding_dash_count = len(stripped)
            current = []
            current_start = pos + len(line)
            pos += len(line)
            continue
        current.append(line)
        pos += len(line)

    if not saw_separator:
        return [(text, 0, 0)] if with_offsets else [(text, 0)]
    raw_tail = "".join(current)
    tail = raw_tail.strip()
    if tail:
        lead = len(raw_tail) - len(raw_tail.lstrip())
        parts.append((tail, preceding_dash_count, current_start + lead))
    if not parts:
        return [(text, 0, 0)] if with_offsets else [(text, 0)]
    if len(parts) > max_parts:
        head = parts[: max_parts - 1]
        overflow = parts[max_parts - 1 :]
        merged = "\n\n".join(part for part, _, _ in overflow)
        # The merged tail begins where the first overflow part begins.
        parts = head + [(merged, overflow[0][1], overflow[0][2])]
    if with_offsets:
        return parts
    return [(part, dash) for part, dash, _ in parts]


def split_reply_delay_seconds(base_delay: float, dash_count: int) -> float:
    """Scale pause after --- by dash count. --- is base, ---- is 1.5x, etc."""
    if dash_count < 3:
        return base_delay
    return base_delay * min(5.0, 1.0 + (dash_count - 3) * 0.5)

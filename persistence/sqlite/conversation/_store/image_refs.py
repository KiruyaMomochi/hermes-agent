"""Ordered image sidecars, separate from the searchable text projection."""
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from session_rules.turns import _IMAGE_PART_TYPES, _is_multimodal_tool_result, _durable_content
from persistence.sqlite._state.images import retain_image, load_image

logger = logging.getLogger(__name__)


def _image_source(part: dict) -> Any:
    url = part.get("image_url")
    if url:
        return url.get("url") if isinstance(url, dict) else url
    source = part.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    return source.get("url")


def _retain_part(part: dict, db_path: str) -> dict:
    url = _image_source(part)
    if not isinstance(url, str):
        raise ValueError("missing image source")
    if urlparse(url).scheme in {"http", "https"}:
        return {"type": "image_ref", "url": url}
    if url.startswith("data:"):
        header, encoded = url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("non-base64 data URL")
        image = base64.b64decode(encoded, validate=True)
        mime = header[5:].split(";", 1)[0]
    else:
        path = Path(unquote(urlparse(url).path)) if url.startswith("file://") else Path(url)
        image = path.read_bytes()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.lower(), "image/png")
    if not image:
        raise ValueError("empty image")
    return {"type": "image_url", "sha256": retain_image(image, db_path), "mime": mime}


def _durable_image_parts(content: Any, db_path: str) -> Optional[str]:
    """Snapshot local/data sources; HTTP references remain URLs, never fetched on replay."""
    parts = content.get("content") if _is_multimodal_tool_result(content) else content
    if not isinstance(parts, list) or not any(isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES for p in parts):
        return None
    saved = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            saved.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") in _IMAGE_PART_TYPES:
            try:
                retained = _retain_part(part, db_path)
            except (OSError, ValueError) as exc:
                logger.warning("Could not retain session image: %s", exc)
                retained = {"type": "image_unavailable"}
            image_url = part.get("image_url")
            detail = image_url.get("detail") if isinstance(image_url, dict) else part.get("detail")
            if detail in ("low", "high", "auto"):
                retained["detail"] = detail
            saved.append(retained)
    return json.dumps(saved)


def _restore_part(part: dict, db_path: str) -> dict:
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": str(part.get("text", ""))}
    try:
        if kind == "image_ref":
            url = part.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError("invalid remote image")
        elif kind == "image_url":
            digest = part.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid image digest")
            image = load_image(digest, db_path)
            mime = part.get("mime") if part.get("mime") in {"image/png", "image/jpeg", "image/webp", "image/gif"} else "image/png"
            url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        else:
            raise ValueError("unavailable image")
    except (OSError, ValueError):
        return {"type": "text", "text": "[image unavailable]"}
    image_url = {"url": url}
    if part.get("detail") in ("low", "high", "auto"):
        image_url["detail"] = part["detail"]
    return {"type": "image_url", "image_url": image_url}


def _restore_image_parts(raw: str, db_path: str) -> list[dict]:
    """Fail closed on absent or damaged snapshots, including stale native-visibility claims."""
    try:
        parts = json.loads(raw)
    except (ValueError, TypeError):
        parts = None
    if not isinstance(parts, list):
        return [{"type": "text", "text": "[image unavailable]"}]
    restored = [_restore_part(part, db_path) for part in parts if isinstance(part, dict)]
    if any(part == {"type": "text", "text": "[image unavailable]"} for part in restored):
        for part in restored:
            if part.get("type") == "text":
                part["text"] = re.sub(r"you can see it natively now", "image unavailable", part["text"], flags=re.IGNORECASE)
    return restored


def image_content(content: Any, db_path: str) -> tuple[Any, Optional[str]]:
    """Regenerate from authoritative content, clearing stale sidecars on image removal."""
    refs = _durable_image_parts(content, db_path)
    projected = _durable_content(content) if refs or _is_multimodal_tool_result(content) else content
    return projected, refs


def restored_content(row, db_path: str, decode_content_fn) -> Any:
    """The replay body, including the pre-sidecar vision fallback."""
    if "image_refs" in row.keys() and row["image_refs"]:
        return _restore_image_parts(row["image_refs"], db_path)
    content = decode_content_fn(row["content"])
    if isinstance(content, str) and "tool_name" in row.keys() and row["role"] == "tool" and row["tool_name"] == "vision_analyze":
        content = content.replace("[screenshot]", "[image unavailable]")
        content = re.sub(r"you can see it natively now", "image unavailable", content, flags=re.IGNORECASE)
    return content

"""Content-addressed image snapshots under the owning database's home."""
import hashlib
from pathlib import Path


def retain_image(image: bytes, db_path: str) -> str:
    digest = hashlib.sha256(image).hexdigest()
    store = Path(db_path).parent / "cache" / "session_images"
    store.mkdir(parents=True, exist_ok=True)
    target = store / digest
    if not target.exists():
        target.write_bytes(image)
    return digest


def load_image(digest: str, db_path: str) -> bytes:
    image = (Path(db_path).parent / "cache" / "session_images" / digest).read_bytes()
    if hashlib.sha256(image).hexdigest() != digest:
        raise ValueError("image snapshot hash mismatch")
    return image

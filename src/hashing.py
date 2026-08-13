import hashlib
from pathlib import Path


def calculate_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

"""Small helpers: logging, hashing, disk space, sizes."""
import hashlib
import logging
import shutil
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def sha256_of(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks to keep memory use low."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def free_space_bytes(path: Path) -> int:
    """Free bytes on the disk that holds path."""
    path = Path(path)
    # The folder may not exist yet: walk up to the nearest existing parent
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def human_size(num_bytes: float) -> str:
    """Human-readable size, e.g. 6885481608 -> '6.4 GiB'."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PiB"
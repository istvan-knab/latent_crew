"""ZIP/MAT extraction helpers."""
import zipfile
from pathlib import Path

from tqdm import tqdm

from .validators import ValidationError


def safe_extract_zip(archive: Path, destination: Path) -> int:
    """Extract a ZIP, refusing any member that would land outside destination."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        # Check all paths before writing anything (path traversal guard)
        for m in members:
            target = (destination / m.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValidationError(f"Unsafe path in ZIP: {m.filename}")
        for m in tqdm(members, desc=f"Extracting {Path(archive).name}", unit="file"):
            zf.extract(m, destination)
    return len(members)
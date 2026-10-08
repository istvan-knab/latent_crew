"""ZIP/MAT extraction helpers."""
import zipfile
from pathlib import Path
import h5py
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

def h5_strings(f: h5py.File, key: str) -> list[str]:
    """Decode a MATLAB cell array of strings (v7.3 stores them as references)."""
    return ["".join(map(chr, f[ref][()].ravel())) for ref in f[key][()].ravel()]


def read_nyu_sample(f: h5py.File, i: int) -> dict:
    """Load NYU sample i as arrays in (H, W) / (H, W, 3) layout."""
    # MATLAB v7.3 stores arrays transposed, so swap the axes back
    return {
        "rgb": f["images"][i].transpose(2, 1, 0),
        "depth": f["depths"][i].T,
        "semantic": f["labels"][i].T,
        "instance": f["instances"][i].T,
    }
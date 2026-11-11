"""Makes sure the processed dataset exists before training; downloads only when something is missing."""

import json
import subprocess
import sys
from pathlib import Path

# Files the segmentation Dataset reads; the rest of index.jsonl (instance, depth_raw) is not needed
REQUIRED_FIELDS = ("rgb", "depth", "semantic")


def count_missing_files(root: Path) -> int | None:
    """None if there is no index.jsonl yet, otherwise the number of required files that do not exist."""
    index = root / "index.jsonl"
    if not index.exists() or not (root / "classes.json").exists():
        return None
    missing = 0
    with open(index, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                missing += sum(not (root / row[k]).exists() for k in REQUIRED_FIELDS)
    return missing


def ensure_dataset(name: str, root: str | Path) -> None:
    """Uses the processed data if it is complete; otherwise runs data/scripts/collect_data.py for this dataset.

    The collector itself is idempotent and resumable: it only downloads, extracts and converts what is missing.
    """
    root = Path(root)
    missing = count_missing_files(root)
    if missing == 0:
        print(f"Dataset {name} found in {root}, using it (no download)")
        return

    # The collector writes to <data_dir>/processed/<name>, so the config root must follow that layout
    if root.name != name or root.parent.name != "processed":
        raise ValueError(f"dataset.root must be <data dir>/processed/{name} to be filled automatically, got {root}")
    data_dir = root.parent.parent
    reason = "no index.jsonl" if missing is None else f"{missing} files missing"
    print(f"Dataset {name} incomplete in {root} ({reason}): running the data collector")
    subprocess.run(
        [sys.executable, str(data_dir / "scripts" / "collect_data.py"), "--datasets", name, "--data-root", str(data_dir)],
        check=True,
    )

    missing = count_missing_files(root)
    if missing != 0:
        raise RuntimeError(f"Dataset {name} is still incomplete after collecting: {missing} files missing")

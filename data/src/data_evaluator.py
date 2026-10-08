"""DataEvaluator: statistics and quality report for the prepared datasets."""
import json
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .data_collector import PATH_KEYS
from .utils import get_logger, human_size
from .validators import ValidationError

log = get_logger(__name__)

# Fixed seed, so --sample-limit picks the same samples on every run
SEED = 0
# PIL image mode -> numpy dtype of the pixel values
MODE_DTYPES = {"RGB": "uint8", "L": "uint8", "I;16": "uint16", "I": "int32", "F": "float32"}


class DataEvaluator:
    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.cfg = load_config(self.data_root)
        with open(self.cfg.manifest, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.sample_limit: int | None = None
        self.problems: list[dict] = []
        self._indexes: dict[str, list[dict]] = {}

    def evaluate(self, datasets: list[str], sample_limit: int | None = None) -> dict:
        """Run every check on each dataset and collect the results."""
        self.sample_limit = sample_limit
        self.problems = []
        start = time.perf_counter()
        result = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "program": self._program_info(),
            "sample_limit": sample_limit,
            "datasets": {},
        }
        for name in datasets:
            log.info("Evaluating %s", name)
            result["datasets"][name] = {
                "structure": self.evaluate_structure(name),
                "rgb": self.evaluate_rgb(name),
                "depth": self.evaluate_depth(name),
                "labels": self.evaluate_labels(name),
                "alignment": self.evaluate_alignment(name),
            }
        result["problematic_samples"] = len(self.problems)
        result["runtime_s"] = round(time.perf_counter() - start, 1)
        return result

    def _program_info(self) -> dict:
        """Code and library versions, so a report can be traced back to its source."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=self.data_root,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        return {"git_commit": commit, "python": platform.python_version(),
                "numpy": np.__version__, "pillow": PIL.__version__}

    def _load_index(self, dataset: str) -> list[dict]:
        """Rows of processed/<dataset>/index.jsonl (cached)."""
        if dataset not in self._indexes:
            index = self.cfg.processed / dataset / "index.jsonl"
            if not index.exists():
                raise ValidationError(
                    f"Missing {index}, run collect_data.py --prepare first")
            with open(index, encoding="utf-8") as fh:
                self._indexes[dataset] = [json.loads(line) for line in fh]
        return self._indexes[dataset]

    def _subset(self, rows: list[dict]) -> list[dict]:
        """At most sample_limit rows, picked at random but reproducibly."""
        if self.sample_limit is None or self.sample_limit >= len(rows):
            return rows
        picked = random.Random(SEED).sample(range(len(rows)), self.sample_limit)
        return [rows[i] for i in sorted(picked)]

    def _problem(self, dataset: str, sample_id: str, check: str, message: str) -> None:
        self.problems.append(
            {"dataset": dataset, "id": sample_id, "check": check, "message": message})

    def evaluate_structure(self, dataset: str) -> dict:
        """File counts, splits, missing or unreadable files, formats and resolutions."""
        start = time.perf_counter()
        out = self.cfg.processed / dataset
        rows = self._load_index(dataset)
        spec = self.manifest[dataset]
        n = len(rows)

        # Index consistency, checked on the full index
        splits = Counter(r["split"] for r in rows)
        ids = Counter(r["id"] for r in rows)
        paths = Counter(r[k] for r in rows for k in PATH_KEYS if k in r)
        unknown_split = [r["id"] for r in rows if r["split"] not in ("train", "test")]

        missing, incomplete = Counter(), 0
        for r in tqdm(rows, desc=f"Checking {dataset} files", unit="sample"):
            gone = [k for k in PATH_KEYS if k in r and not (out / r[k]).exists()]
            missing.update(gone)
            if gone:
                incomplete += 1
                self._problem(dataset, r["id"], "structure", f"missing files: {gone}")

        # Formats from the file headers only (pixels are decoded in the other checks)
        subset = self._subset(rows)
        files = {}
        unreadable = Counter()
        for r in tqdm(subset, desc=f"Reading {dataset} headers", unit="sample"):
            for k in PATH_KEYS:
                if k not in r or not (out / r[k]).exists():
                    continue
                stats = files.setdefault(
                    k, {"format": Counter(), "dtype": Counter(), "resolution": Counter()})
                try:
                    with Image.open(out / r[k]) as img:
                        stats["format"][img.format] += 1
                        stats["dtype"][MODE_DTYPES.get(img.mode, img.mode)] += 1
                        stats["resolution"][f"{img.width}x{img.height}"] += 1
                except OSError as e:
                    unreadable[k] += 1
                    self._problem(dataset, r["id"], "structure", f"unreadable {k}: {e}")

        downloads = sum(f["size_bytes"] for f in spec["files"])
        processed = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        return {
            "name": spec["name"],
            "version": spec["version"],
            "source_files": [
                {"file": Path(f["download"]).name, "size_bytes": f["size_bytes"],
                 "sha256": f["sha256"]}
                for f in spec["files"]
            ],
            "disk_usage": {
                "downloads_bytes": downloads, "downloads": human_size(downloads),
                "processed_bytes": processed, "processed": human_size(processed),
            },
            "samples": {
                "total": n, "train": splits["train"], "test": splits["test"],
                "train_ratio": round(splits["train"] / n, 3) if n else None,
            },
            "index_issues": {
                "duplicate_ids": sum(c > 1 for c in ids.values()),
                "duplicate_paths": sum(c > 1 for c in paths.values()),
                "unknown_split": len(unknown_split),
            },
            "missing_files": dict(missing),
            "incomplete_samples": incomplete,
            "checked_samples": len(subset),
            "files": files,
            "unreadable_files": dict(unreadable),
            "runtime_s": round(time.perf_counter() - start, 1),
        }

    def evaluate_rgb(self, dataset: str) -> dict:
        return {}  # TODO

    def evaluate_depth(self, dataset: str) -> dict:
        return {}  # TODO

    def evaluate_labels(self, dataset: str) -> dict:
        return {}  # TODO

    def evaluate_alignment(self, dataset: str) -> dict:
        return {}  # TODO

    def write_reports(self, result: dict, output_dir: Path | None = None) -> None:
        """Write the full JSON report and a short Markdown summary."""
        out = Path(output_dir) if output_dir else self.cfg.reports
        out.mkdir(parents=True, exist_ok=True)
        (out / "data_evaluation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        lines = ["# Data evaluation", "", f"Created: {result['created']}", ""]
        for name, parts in result["datasets"].items():
            lines.append(f"## {name}")
            for part, values in parts.items():
                lines.append(f"- **{part}**: {len(values)} metrics")
            lines.append("")
        (out / "data_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
        log.info("Reports written to %s", out)

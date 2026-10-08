"""DataEvaluator: statistics and quality report for the prepared datasets."""
import json
import time
from datetime import datetime
from pathlib import Path

from .config import load_config
from .utils import get_logger

log = get_logger(__name__)


class DataEvaluator:
    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.cfg = load_config(self.data_root)

    def evaluate(self, datasets: list[str], sample_limit: int | None = None) -> dict:
        """Run every check on each dataset and collect the results."""
        self.sample_limit = sample_limit
        start = time.perf_counter()
        result = {
            "created": datetime.now().isoformat(timespec="seconds"),
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
        result["runtime_s"] = round(time.perf_counter() - start, 1)
        return result

    def evaluate_structure(self, dataset: str) -> dict:
        return {}  # TODO

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
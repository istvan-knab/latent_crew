"""CLI entry point: evaluate the prepared datasets and write reports."""
import argparse
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_ROOT))  # make `src` importable from anywhere

from src.data_evaluator import DataEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["sunrgbd", "nyu_depth_v2"])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    evaluator = DataEvaluator(args.data_root)
    result = evaluator.evaluate(args.datasets, sample_limit=args.sample_limit)
    evaluator.write_reports(result, args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
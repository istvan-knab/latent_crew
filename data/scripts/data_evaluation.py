"""CLI entry point: evaluate the prepared datasets and write reports."""
import argparse
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_ROOT))  # make `src` importable from anywhere

from src.data_evaluator import DataEvaluator  # noqa: E402
from src.validators import ValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["sunrgbd", "nyu_depth_v2"])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    try:
        evaluator = DataEvaluator(args.data_root)
        result = evaluator.evaluate(args.datasets, sample_limit=args.sample_limit)
        evaluator.write_reports(result, args.output_dir)
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    for name, s in result["summary"].items():
        print(f"{name:<14}{s['samples']:>6} samples ({s['train']} train / {s['test']} test)"
              f"  flagged: {s['flagged_samples']} (errors: {s['errors']},"
              f" warnings: {s['warnings']})")
    out = args.output_dir or evaluator.cfg.reports
    print(f"Reports: {out / 'data_evaluation.md'}, data_evaluation.json, "
          f"problematic_samples.jsonl")
    if result["errors"]:
        print(f"ERROR: {result['errors']} samples with errors, see problematic_samples.jsonl",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

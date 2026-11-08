"""CLI entry point: download, verify and prepare the datasets."""
import argparse
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_ROOT))  # make `src` importable from anywhere

from src.data_collector import DataCollector  # noqa: E402
from src.validators import ValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="default: from config.toml")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--boxes3d", action="store_true",
                        help="only (re)write boxes3d.jsonl of an already prepared SUN RGB-D")
    parser.add_argument("--resplit", action="store_true",
                        help="only redo the train/val split in index.jsonl (fast, no file is touched)")
    parser.add_argument("--val-ratio", type=float,
                        help="fraction of the official train set used for validation "
                             "(default: [split] val_ratio in config.toml)")
    parser.add_argument("--seed", type=int, help="split seed (default: [split] seed)")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-raw-nyu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    collector = DataCollector(args.data_root)
    datasets = args.datasets or collector.cfg.datasets
    # No step flag given: run every step
    run_all = not (args.verify or args.extract or args.prepare)

    # The split settings apply to --prepare and --resplit
    if args.val_ratio is not None:
        collector.cfg.val_ratio = args.val_ratio
    if args.seed is not None:
        collector.cfg.seed = args.seed

    try:
        if args.resplit or args.boxes3d:
            for name in datasets:
                if args.resplit:
                    collector.resplit(name)
                if args.boxes3d:
                    collector.export_boxes3d(name)
        elif args.verify_only:
            for name in datasets:
                for spec in collector.manifest[name]["files"]:
                    collector.verify_download(collector.cfg.downloads / spec["download"], spec)
                collector.verify_dataset(name)
        else:
            collector.collect(
                datasets,
                verify=args.verify or run_all,
                extract=args.extract or run_all,
                prepare=args.prepare or run_all,
                include_raw_nyu=args.include_raw_nyu,
                force=args.force,
            )
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
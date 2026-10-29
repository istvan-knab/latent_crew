"""Evaluates a trained detection checkpoint."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a trained detection checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()

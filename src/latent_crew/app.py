"""Runs the full detection + segmentation app end-to-end."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the full detection + segmentation app end-to-end.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--detection-checkpoint", required=True, help="Path to the detection checkpoint")
    parser.add_argument("--segmentation-checkpoint", required=True, help="Path to the segmentation checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()

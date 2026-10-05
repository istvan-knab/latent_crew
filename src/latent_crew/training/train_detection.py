"""Trains the 3D object detection model."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains the 3D object detection model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()

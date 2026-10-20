"""Shows one segmentation sample as the model sees it: RGB, depth, label mask and overlay side by side.

Usage:
    python -m latent_crew.utils.visualization --config configs/segmentation.yaml --index 0
    python -m latent_crew.utils.visualization --config configs/segmentation.yaml --id sun_000123 --augment
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # renders to file, also inside Docker without a display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from latent_crew.config import SegmentationConfig
from latent_crew.data.segmentation_dataset import SegmentationDataset

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)
MAX_LEGEND_CLASSES = 12


def class_palette(num_classes: int) -> np.ndarray:
    """Fixed colour per class (same class = same colour in every image), uint8 [num_classes, 3]."""
    colors = [plt.get_cmap(name)(i)[:3] for name in ("tab20", "tab20b", "tab20c") for i in range(20)]
    return (np.array(colors[:num_classes]) * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, palette: np.ndarray, ignore_index: int) -> np.ndarray:
    """Class ids [H, W] -> RGB [H, W, 3]; ignored pixels are black."""
    color = np.empty((*mask.shape, 3), dtype=np.uint8)
    ignored = mask == ignore_index
    color[ignored] = IGNORE_COLOR
    color[~ignored] = palette[mask[~ignored]]
    return color


def render_sample(
    views: Dict[str, np.ndarray],
    mask: np.ndarray,
    class_names: List[str],
    ignore_index: int,
    title: str,
    prediction: np.ndarray | None = None,
) -> Figure:
    """views: {"rgb": uint8 [H, W, 3], "depth": mm [H, W]} (either may be missing); mask: class ids [H, W].

    prediction: predicted class ids [H, W]; adds a panel, shown only where the label is not ignored.
    """
    palette = class_palette(len(class_names))
    mask_rgb = colorize_mask(mask, palette, ignore_index)

    panels = []
    if "rgb" in views:
        panels.append(("RGB", views["rgb"]))
    if "depth" in views:
        panels.append(("Depth (m)", views["depth"] / 1000.0))
    panels.append(("Label mask (black = ignored)", mask_rgb))
    if prediction is not None:
        shown_prediction = np.where(mask == ignore_index, ignore_index, prediction)
        panels.append(("Prediction", colorize_mask(shown_prediction, palette, ignore_index)))
    if "rgb" in views:
        ignored = (mask == ignore_index)[..., None]
        overlay = np.where(ignored, views["rgb"], (0.5 * views["rgb"] + 0.5 * mask_rgb).astype(np.uint8))
        panels.append(("Overlay", overlay))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.6), layout="constrained")
    for ax, (name, image) in zip(axes, panels):
        if image.ndim == 2:
            shown = ax.imshow(image, cmap="turbo")
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")

    # Legend: the classes covering the most pixels in this sample
    ids, counts = np.unique(mask[mask != ignore_index], return_counts=True)
    top = ids[np.argsort(counts)[::-1][:MAX_LEGEND_CLASSES]]
    total = counts.sum()
    share = dict(zip(ids, counts))
    handles = [Patch(color=palette[k] / 255.0, label=f"{class_names[k]} ({share[k] / total:.0%})") for k in top]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 6), fontsize=9, frameon=False)
    fig.suptitle(title)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shows one segmentation sample: RGB, depth, mask, overlay.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--index", type=int, default=0, help="Position within the split")
    which.add_argument("--id", help="Sample id from index.jsonl, e.g. sun_000123")
    parser.add_argument("--augment", action="store_true", help="Apply the training augmentation (random each run)")
    parser.add_argument("--out", help="Output PNG (default: outputs/visualizations/<id>[_aug].png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SegmentationConfig.load_from_yaml(args.config)
    dataset = SegmentationDataset.from_config(config, args.split, augment=args.augment)

    index = args.index
    if args.id is not None:
        ids = [s["id"] for s in dataset.samples]
        if args.id not in ids:
            raise SystemExit(f"{args.id} is not in the {args.split} split")
        index = ids.index(args.id)

    sample = dataset[index]
    views = dataset.denormalize(sample["image"])
    mask = sample["mask"].numpy()
    title = f"{sample['id']} ({args.split}, {'augmented' if args.augment else 'no augmentation'}, {tuple(mask.shape)})"
    fig = render_sample(views, mask, dataset.class_names, config.dataset.ignore_index, title)

    out = Path(args.out or f"outputs/visualizations/{sample['id']}{'_aug' if args.augment else ''}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

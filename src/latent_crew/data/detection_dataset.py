"""Builds a YOLO-format 2D detection dataset from the processed SUN RGB-D data and its box metadata.

SUN RGB-D is the only dataset here with 2D boxes: ``SUNRGBDMeta2DBB_v2.mat`` holds, per image, a list of
boxes with a ``classname`` and ``gtBb2D`` (x, y, w, h in pixels, top-left origin). NYU Depth V2 has no 2D
box annotation, so it is not used for detection.

The processed ``index.jsonl`` gives the RGB image path, the train/val/test split and the ``source`` folder
of each SUN sample; ``source`` is matched to the metadata through ``sun_key`` (the same key the data pipeline
uses). This module writes a standard YOLO layout:

    <yolo_root>/
      data.yaml                 # paths, class names, nc
      classes.json              # {"names": [...], "counts": {...}}  (for reproducibility / docs)
      images/{train,val,test}/  # RGB images (copied from the processed data)
      labels/{train,val,test}/  # one .txt per image: "cls cx cy w h" normalised to [0, 1]

Ultralytics maps val -> validation and test -> an optional held-out split, both referenced from data.yaml.
"""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import yaml
from PIL import Image

# Must match the keys used by data/src/extractors.py::sun_key
_SUN_ANCHOR = "SUNRGBD/"
_META_FILE = "SUNRGBDMeta2DBB_v2.mat"
_META_VAR = "SUNRGBDMeta2DBB"
_META_FIELD = "groundtruth2DBB"
_SPLITS = ("train", "val", "test")


def sun_key(path: str) -> str:
    """'/n/fs/sun3d/data/SUNRGBD/kv1/x/' -> 'SUNRGBD/kv1/x' (shared key for metadata and the index)."""
    path = path.rstrip("/")
    return path[path.index(_SUN_ANCHOR):]


def _read_sun_2d_boxes(meta_path: Path) -> Dict[str, List[Tuple[str, float, float, float, float]]]:
    """key -> list of (classname, x, y, w, h) in pixels, read from SUNRGBDMeta2DBB_v2.mat."""
    mat = scipy.io.loadmat(meta_path, variable_names=[_META_VAR], squeeze_me=True, struct_as_record=False)
    boxes_by_key: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    for m in np.atleast_1d(mat[_META_VAR]):
        key = sun_key(m.sequenceName)
        boxes = []
        for b in np.atleast_1d(getattr(m, _META_FIELD)):
            if not hasattr(b, "classname"):
                continue  # samples without boxes hold an empty array instead of box structs
            x, y, w, h = np.asarray(b.gtBb2D, np.float64).ravel()[:4]
            if np.isfinite([x, y, w, h]).all() and w > 0 and h > 0:
                boxes.append((str(b.classname), float(x), float(y), float(w), float(h)))
        boxes_by_key[key] = boxes
    return boxes_by_key


def _load_index(root: Path) -> List[dict]:
    """Reads processed/<name>/index.jsonl (one JSON object per line)."""
    rows = []
    with open(root / "index.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _choose_classes(
    rows: List[dict], boxes_by_key: Dict[str, list], num_classes: int, min_box_size: int
) -> Dict[str, int]:
    """The ``num_classes`` most frequent class names (over all splits), mapped to contiguous ids 0..N-1."""
    counts: Counter = Counter()
    for row in rows:
        for name, _x, _y, w, h in boxes_by_key.get(sun_key(row["source"]), []):
            if w >= min_box_size and h >= min_box_size:
                counts[name] += 1
    top = [name for name, _ in counts.most_common(num_classes)]
    return {name: i for i, name in enumerate(top)}


def _to_yolo_lines(
    boxes: List[Tuple[str, float, float, float, float]],
    class_to_id: Dict[str, int],
    img_w: int,
    img_h: int,
    min_box_size: int,
) -> List[str]:
    """Converts pixel (x, y, w, h) top-left boxes to normalised YOLO 'cls cx cy w h' lines, clipped to the image."""
    lines = []
    for name, x, y, w, h in boxes:
        if name not in class_to_id or w < min_box_size or h < min_box_size:
            continue
        # Clip to the image, then take the normalised centre and size of the clipped box
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
        bw, bh = x1 - x0, y1 - y0
        if bw < min_box_size or bh < min_box_size:
            continue
        cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
        lines.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw / img_w:.6f} {bh / img_h:.6f}")
    return lines


def build_yolo_dataset(
    processed_root: str | Path,
    raw_root: str | Path,
    yolo_root: str | Path,
    num_classes: int,
    min_box_size: int = 4,
    force: bool = False,
) -> Path:
    """Builds (or reuses) the YOLO dataset and returns the path of its ``data.yaml``.

    Idempotent: if ``data.yaml`` already exists and ``force`` is False, the existing dataset is reused.
    """
    processed_root, raw_root, yolo_root = Path(processed_root), Path(raw_root), Path(yolo_root)
    data_yaml = yolo_root / "data.yaml"
    if data_yaml.exists() and not force:
        print(f"YOLO dataset found at {yolo_root}, reusing it (data.yaml exists)")
        return data_yaml

    meta_path = raw_root / "sunrgbd" / "metadata" / _META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"SUN RGB-D 2D box metadata not found at {meta_path}. "
            "Run the data collector first (python data/scripts/collect_data.py --datasets sunrgbd)."
        )

    print(f"Reading 2D boxes from {meta_path}")
    boxes_by_key = _read_sun_2d_boxes(meta_path)
    rows = _load_index(processed_root)
    class_to_id = _choose_classes(rows, boxes_by_key, num_classes, min_box_size)
    if not class_to_id:
        raise RuntimeError("No boxes found for any class; check the metadata and index paths")
    names = sorted(class_to_id, key=class_to_id.get)
    print(f"Keeping {len(names)} classes: {names}")

    if yolo_root.exists() and force:
        shutil.rmtree(yolo_root)
    for split in _SPLITS:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept_counts: Counter = Counter()
    per_split: Counter = Counter()
    images_without_boxes = 0
    for row in rows:
        split = row["split"]
        src_rgb = processed_root / row["rgb"]
        if not src_rgb.exists():
            continue
        with Image.open(src_rgb) as img:
            img_w, img_h = img.size
        boxes = boxes_by_key.get(sun_key(row["source"]), [])
        lines = _to_yolo_lines(boxes, class_to_id, img_w, img_h, min_box_size)
        if not lines:
            images_without_boxes += 1
            continue  # YOLO ignores images with no label file; skip to keep the set clean
        stem = row["id"]  # e.g. sun_000123 -> unique across splits
        shutil.copy2(src_rgb, yolo_root / "images" / split / f"{stem}{src_rgb.suffix}")
        (yolo_root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        per_split[split] += 1
        for line in lines:
            kept_counts[names[int(line.split()[0])]] += 1

    data = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with open(yolo_root / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"names": names, "counts": dict(kept_counts.most_common())}, f, indent=2)

    print(
        f"Built YOLO dataset at {yolo_root}: "
        f"train={per_split['train']} val={per_split['val']} test={per_split['test']} images, "
        f"{sum(kept_counts.values())} boxes, {images_without_boxes} images had no kept box (skipped)"
    )
    return data_yaml

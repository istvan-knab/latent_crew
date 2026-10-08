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
import scipy.io
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .data_collector import PATH_KEYS
from .extractors import sun_key
from .utils import get_logger, human_size
from .validators import ValidationError

log = get_logger(__name__)

# Fixed seed, so --sample-limit picks the same samples on every run
SEED = 0
# PIL image mode -> numpy dtype of the pixel values
MODE_DTYPES = {"RGB": "uint8", "L": "uint8", "I;16": "uint16", "I": "int32", "F": "float32"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
# Thresholds for suspicious RGB images, on the 0-1 scale
RGB_DARK_MEAN = 0.1
RGB_BRIGHT_MEAN = 0.9
RGB_LOW_STD = 0.02
# Depth PNGs: uint16 millimetres, 0 = no measurement
DEPTH_NEAR_MM = 100     # closer than 0.1 m is physically implausible for these sensors
DEPTH_FAR_MM = 10000    # beyond 10 m is outside the indoor sensor range
DEPTH_MIN_VALID = 0.5   # flag samples whose raw depth has fewer measured pixels than this
DEPTH_DIFF_MM = 10      # raw vs. filled differences above this count as changed
# A class is rare below this share of labelled pixels or in fewer images than this
RARE_PIXEL_RATIO = 0.001
RARE_IMAGES = 10
TOP_CLASSES = 10
# Sample ids listed per issue in the report (all of them go to self.problems)
EXAMPLES = 10


def _hist_stats(hist: np.ndarray, scale: float = 255, digits: int = 4) -> dict:
    """Exact statistics from a histogram whose bin i holds the count of value i.

    Values are divided by scale: 255 gives the 0-1 range for uint8, 1 keeps raw values.
    """
    n = hist.sum()
    if n == 0:
        return {}
    v = np.arange(len(hist)) / scale
    mean = (v * hist).sum() / n
    std = np.sqrt(((v - mean) ** 2 * hist).sum() / n)
    nonzero = np.flatnonzero(hist)
    cdf = np.cumsum(hist) / n

    def pct(p: float) -> float:
        return round(int(np.searchsorted(cdf, p / 100)) / scale, digits)

    return {
        "mean": round(float(mean), digits),
        "std": round(float(std), digits),
        "min": round(float(nonzero[0]) / scale, digits),
        "max": round(float(nonzero[-1]) / scale, digits),
        "median": pct(50),
        "percentiles": {str(p): pct(p) for p in PERCENTILES},
    }


def _summary(values: list[float]) -> dict:
    """Min, mean and max of a list of per-sample values."""
    if not values:
        return {}
    return {"min": round(float(min(values)), 4), "mean": round(float(np.mean(values)), 4),
            "max": round(float(max(values)), 4)}


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
        """Channel statistics and dark, bright, low-contrast or broken RGB images."""
        start = time.perf_counter()
        out = self.cfg.processed / dataset
        rows = self._subset(self._load_index(dataset))

        # Per-sensor histograms: 3 channels x 256 values, exact for uint8 images
        hists: dict[str, np.ndarray] = {}
        sensor_samples = Counter()
        brightness = []
        flagged = {k: [] for k in ("dark", "bright", "low_contrast", "bad_channels",
                                   "unreadable")}

        def flag(kind: str, sample_id: str, message: str) -> None:
            flagged[kind].append(sample_id)
            self._problem(dataset, sample_id, "rgb", message)

        for r in tqdm(rows, desc=f"RGB {dataset}", unit="sample"):
            try:
                with Image.open(out / r["rgb"]) as img:
                    a = np.asarray(img)
            except OSError as e:
                flag("unreadable", r["id"], f"unreadable rgb: {e}")
                continue
            if a.ndim != 3 or a.shape[2] != 3:
                flag("bad_channels", r["id"], f"rgb shape {a.shape}, expected (H, W, 3)")
                continue

            px = a.reshape(-1, 3)
            h = hists.setdefault(r["sensor"], np.zeros((3, 256), np.int64))
            for c in range(3):
                h[c] += np.bincount(px[:, c], minlength=256)
            sensor_samples[r["sensor"]] += 1

            mean, std = px.mean() / 255, px.std() / 255
            brightness.append(mean)
            if mean < RGB_DARK_MEAN:
                flag("dark", r["id"], f"dark image (mean {mean:.3f})")
            elif mean > RGB_BRIGHT_MEAN:
                flag("bright", r["id"], f"bright image (mean {mean:.3f})")
            if std < RGB_LOW_STD:
                flag("low_contrast", r["id"], f"low contrast image (std {std:.3f})")

        total = sum(hists.values()) if hists else np.zeros((3, 256), np.int64)
        by_sensor = {}
        for sensor, h in sorted(hists.items()):
            stats = [_hist_stats(h[c]) for c in range(3)]
            by_sensor[sensor] = {
                "samples": sensor_samples[sensor],
                "mean": [s["mean"] for s in stats],
                "std": [s["std"] for s in stats],
            }
        return {
            "checked_samples": len(rows),
            "scale": "0-1 (pixel value / 255)",
            "thresholds": {"dark_mean": RGB_DARK_MEAN, "bright_mean": RGB_BRIGHT_MEAN,
                           "low_contrast_std": RGB_LOW_STD},
            "channels": {name: _hist_stats(total[c]) for c, name in enumerate("RGB")},
            "brightness": _summary(brightness),
            "by_sensor": by_sensor,
            "flagged": {k: {"count": len(ids), "examples": ids[:EXAMPLES]}
                        for k, ids in flagged.items()},
            "runtime_s": round(time.perf_counter() - start, 1),
        }

    def evaluate_depth(self, dataset: str) -> dict:
        """Depth statistics, invalid and implausible pixels, raw vs. filled depth."""
        start = time.perf_counter()
        out = self.cfg.processed / dataset
        rows = self._subset(self._load_index(dataset))
        fields = [k for k in ("depth", "depth_raw") if rows and k in rows[0]]

        # One histogram per field: bin i counts the pixels at i mm, exact for uint16
        hists = {k: np.zeros(65536, np.int64) for k in fields}
        valid_ratios: dict[str, list[float]] = {k: [] for k in fields}
        # Per sensor, for the filled depth: [sum of valid mm, valid pixels, all pixels]
        sensors: dict[str, np.ndarray] = {}
        # Raw vs. filled: [raw holes, holes filled, both valid, abs diff sum, changed]
        compare = np.zeros(5, np.int64)
        flagged = {k: [] for k in ("unreadable", "bad_format", "empty", "low_valid",
                                   "constant")}

        def flag(kind: str, sample_id: str, message: str) -> None:
            flagged[kind].append(sample_id)
            self._problem(dataset, sample_id, "depth", message)

        for r in tqdm(rows, desc=f"Depth {dataset}", unit="sample"):
            maps = {}
            for k in fields:
                try:
                    with Image.open(out / r[k]) as img:
                        a = np.asarray(img)
                except OSError as e:
                    flag("unreadable", r["id"], f"unreadable {k}: {e}")
                    continue
                if a.dtype != np.uint16 or a.ndim != 2:
                    flag("bad_format", r["id"], f"{k}: {a.dtype} {a.shape}, expected uint16 (H, W)")
                    continue
                maps[k] = a
                hists[k] += np.bincount(a.ravel(), minlength=65536)
                valid_ratios[k].append(float(np.count_nonzero(a)) / a.size)

            d = maps.get("depth")
            if d is not None:
                valid = d > 0
                ratio = valid.mean()
                if ratio == 0:
                    flag("empty", r["id"], "depth has no valid pixels")
                elif d[valid].min() == d[valid].max():
                        flag("constant", r["id"], f"constant depth ({d[valid][0]} mm)")
                s = sensors.setdefault(r["sensor"], np.zeros(3, np.int64))
                s += (int(d[valid].sum(dtype=np.int64)), int(valid.sum()), d.size)

            # Filled depth is always complete, so judge reliability by the measured pixels
            measured = maps.get("depth_raw") if "depth_raw" in fields else d
            if measured is not None:
                ratio = np.count_nonzero(measured) / measured.size
                if 0 < ratio < DEPTH_MIN_VALID:
                    flag("low_valid", r["id"], f"only {ratio:.1%} measured depth pixels")

            raw = maps.get("depth_raw")
            if d is not None and raw is not None and raw.shape == d.shape:
                holes = raw == 0
                both = ~holes & (d > 0)
                diff = np.abs(d[both].astype(np.int32) - raw[both])
                compare += (holes.sum(), (holes & (d > 0)).sum(), both.sum(),
                            diff.sum(dtype=np.int64), (diff > DEPTH_DIFF_MM).sum())

        result = {
            "checked_samples": len(rows),
            "unit": "mm",
            "scale": "uint16 PNG, 1 = 1 mm, 0 = no measurement",
            "thresholds": {"near_mm": DEPTH_NEAR_MM, "far_mm": DEPTH_FAR_MM,
                           "min_valid_ratio": DEPTH_MIN_VALID, "diff_mm": DEPTH_DIFF_MM},
        }
        for k in fields:
            h = hists[k]
            total = h.sum()
            valid_h = h.copy()
            valid_h[0] = 0
            share = (lambda count: round(float(count) / total, 4)) if total else (lambda count: None)
            result[k] = {
                # Statistics over valid (> 0) pixels only
                "valid_pixels": _hist_stats(valid_h, scale=1, digits=1),
                "pixel_ratios": {
                    "zero": share(h[0]),
                    # uint16 PNGs cannot hold negative, NaN or inf values
                    "negative": 0.0, "nan": 0.0, "inf": 0.0,
                    "invalid": share(h[0]),
                    "near": share(h[1:DEPTH_NEAR_MM].sum()),
                    "far": share(h[DEPTH_FAR_MM + 1:].sum()),
                },
                "valid_ratio_per_sample": _summary(valid_ratios[k]),
            }
        result["by_sensor"] = {
            sensor: {"mean_mm": round(s[0] / s[1], 1) if s[1] else None,
                     "valid_ratio": round(s[1] / s[2], 4) if s[2] else None}
            for sensor, s in sorted(sensors.items())
        }
        holes, filled, both, diff_sum, changed = (int(x) for x in compare)
        result["raw_vs_filled"] = {
            "raw_hole_ratio": round(holes / (holes + both), 4) if holes + both else None,
            "holes_filled_ratio": round(filled / holes, 4) if holes else None,
            "mean_abs_diff_mm": round(diff_sum / both, 1) if both else None,
            "changed_ratio": round(changed / both, 4) if both else None,
        } if "depth_raw" in fields else "not available"
        result["flagged"] = {k: {"count": len(ids), "examples": ids[:EXAMPLES]}
                             for k, ids in flagged.items()}
        result["runtime_s"] = round(time.perf_counter() - start, 1)
        return result

    def evaluate_labels(self, dataset: str) -> dict:
        """Class and instance statistics, background ratio, rare classes and boxes."""
        start = time.perf_counter()
        out = self.cfg.processed / dataset
        rows = self._subset(self._load_index(dataset))
        classes = json.loads((out / "classes.json").read_text(encoding="utf-8"))
        n_cls = len(classes)

        pixels = np.zeros(n_cls, np.int64)      # pixels per class
        images = np.zeros(n_cls, np.int64)      # images containing the class
        instances = np.zeros(n_cls, np.int64)   # objects per class
        per_image = {"classes": [], "instances": [], "background_ratio": []}
        sizes = {}  # source folder -> (H, W), for the box checks
        flagged = {k: [] for k in ("unreadable", "bad_format", "bad_values", "unlabeled",
                                   "bad_box_2d", "bad_box_3d")}

        def flag(kind: str, sample_id: str, message: str) -> None:
            flagged[kind].append(sample_id)
            self._problem(dataset, sample_id, "labels", message)

        for r in tqdm(rows, desc=f"Labels {dataset}", unit="sample"):
            maps = {}
            for k in ("semantic", "instance"):
                try:
                    with Image.open(out / r[k]) as img:
                        maps[k] = np.asarray(img)
                except OSError as e:
                    flag("unreadable", r["id"], f"unreadable {k}: {e}")
            sem, inst = maps.get("semantic"), maps.get("instance")
            if sem is None:
                continue
            if sem.ndim != 2 or sem.dtype.kind != "u":
                flag("bad_format", r["id"], f"semantic: {sem.dtype} {sem.shape}")
                continue
            if sem.max() >= n_cls:
                flag("bad_values", r["id"], f"label {sem.max()} >= {n_cls} classes")
                continue

            counts = np.bincount(sem.ravel(), minlength=n_cls)
            pixels += counts
            present = counts > 0
            images += present
            per_image["classes"].append(int(present[1:].sum()))
            per_image["background_ratio"].append(counts[0] / sem.size)
            if counts[0] == sem.size:
                flag("unlabeled", r["id"], "no labelled pixels")
            if "source" in r:
                sizes[r["source"]] = sem.shape

            # An object is one (class, instance id) pair; 0 means none in both maps
            if inst is not None and inst.shape == sem.shape:
                mask = (sem > 0) & (inst > 0)
                pairs = np.unique(sem[mask].astype(np.int64) * 65536 + inst[mask])
                instances += np.bincount(pairs // 65536, minlength=n_cls)
                per_image["instances"].append(len(pairs))

        labelled = int(pixels[1:].sum())
        total = int(pixels.sum())
        per_class = [
            {"id": k, "name": classes[k], "pixels": int(pixels[k]),
             "labelled_ratio": round(int(pixels[k]) / labelled, 6) if labelled else None,
             "images": int(images[k]), "instances": int(instances[k])}
            for k in range(1, n_cls)
        ]
        seen = [c for c in per_class if c["images"] > 0]
        rare = [c["name"] for c in seen
                if c["labelled_ratio"] < RARE_PIXEL_RATIO or c["images"] < RARE_IMAGES]
        by_pixels = sorted(seen, key=lambda c: c["pixels"], reverse=True)
        cum = np.cumsum([c["pixels"] for c in by_pixels]) / labelled if labelled else []

        result = {
            "checked_samples": len(rows),
            "thresholds": {"rare_pixel_ratio": RARE_PIXEL_RATIO, "rare_images": RARE_IMAGES},
            "num_classes": n_cls - 1,
            "class_names": classes,
            "background_ratio": round(int(pixels[0]) / total, 4) if total else None,
            "per_image": {
                "classes": _summary(per_image["classes"]),
                "instances": _summary(per_image["instances"]),
                "background_ratio": _summary(per_image["background_ratio"]),
            },
            "imbalance": {
                "classes_seen": len(seen),
                "classes_never_seen": [c["name"] for c in per_class if c["images"] == 0],
                "rare_classes": rare,
                "top_classes": [{"name": c["name"], "labelled_ratio": c["labelled_ratio"]}
                                for c in by_pixels[:TOP_CLASSES]],
                "max_min_pixel_ratio": round(by_pixels[0]["pixels"] / by_pixels[-1]["pixels"], 1)
                if by_pixels else None,
                "classes_for_90pct_pixels": int(np.searchsorted(cum, 0.9)) + 1 if labelled else None,
            },
            "per_class": per_class,
        }
        if dataset == "sunrgbd":
            result["boxes"] = self._sun_boxes(dataset, rows, sizes, flag)
        else:
            result["boxes"] = "not available"
        # Box issues are per box, so one sample can be listed several times
        result["flagged"] = {k: {"count": len(ids), "examples": list(dict.fromkeys(ids))[:EXAMPLES]}
                             for k, ids in flagged.items()}
        result["runtime_s"] = round(time.perf_counter() - start, 1)
        return result

    def _sun_boxes(self, dataset: str, rows: list[dict], sizes: dict, flag) -> dict:
        """2D and 3D bounding box statistics from the SUN metadata files."""
        meta = self.cfg.raw / "sunrgbd" / "metadata"
        ids = {r["source"]: r["id"] for r in rows}
        result = {}
        for kind, file, var, field in (
            ("2d", "SUNRGBDMeta2DBB_v2.mat", "SUNRGBDMeta2DBB", "groundtruth2DBB"),
            ("3d", "SUNRGBDMeta3DBB_v2.mat", "SUNRGBDMeta", "groundtruth3DBB"),
        ):
            log.info("Reading %s", file)
            mat = scipy.io.loadmat(meta / file, variable_names=[var],
                                   squeeze_me=True, struct_as_record=False)
            per_image, names, dims, bad = [], Counter(), [], 0
            # 2D boxes slightly past the border (a few px) are kept, only counted
            outside, overshoot = 0, 0.0
            for m in np.atleast_1d(mat[var]):
                key = sun_key(m.sequenceName)
                if key not in ids:
                    continue
                boxes = np.atleast_1d(getattr(m, field))
                boxes = [b for b in boxes if hasattr(b, "classname")]
                per_image.append(len(boxes))
                for b in boxes:
                    names[str(b.classname)] += 1
                    if kind == "2d":
                        x, y, w, h = np.asarray(b.gtBb2D, np.float64).ravel()[:4]
                        img_h, img_w = sizes.get(key, (np.inf, np.inf))
                        problem = not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0
                        if not problem:
                            dims.append((w, h, w * h / (img_w * img_h)))
                            past = max(-x, -y, x + w - img_w, y + h - img_h)
                            if past > 1:
                                outside += 1
                                overshoot = max(overshoot, past)
                    else:
                        size = 2 * np.asarray(b.coeffs, np.float64).ravel()
                        centre = np.asarray(b.centroid, np.float64).ravel()
                        problem = (len(size) != 3 or not np.isfinite(size).all()
                                   or (size <= 0).any() or not np.isfinite(centre).all())
                        if not problem:
                            dims.append(tuple(size))
                    if problem:
                        bad += 1
                        flag(f"bad_box_{kind}", ids[key], f"invalid {kind.upper()} box "
                             f"'{b.classname}'")
            dims = np.array(dims) if dims else np.zeros((0, 3))
            labels = (("width_px", "height_px", "area_ratio") if kind == "2d"
                      else ("size_x_m", "size_y_m", "size_z_m"))
            result[kind] = {
                "images": len(per_image),
                "boxes": int(sum(per_image)),
                "per_image": _summary(per_image),
                "invalid": bad,
                **({"outside_image": outside, "max_overshoot_px": round(overshoot, 1)}
                   if kind == "2d" else {}),
                "sizes": {name: _summary(dims[:, i].tolist()) for i, name in enumerate(labels)},
                "distinct_names": len(names),
                "top_names": dict(names.most_common(TOP_CLASSES)),
            }
        return result

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

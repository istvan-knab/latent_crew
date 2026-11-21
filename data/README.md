# Data

Download, verification and preparation of the SUN RGB-D (V1) and NYU Depth V2 (labeled) datasets for 3D object detection and segmentation.

| Dataset | Segmentation (semantic, instance) | 3D object detection (3D bounding boxes) |
|---|---|---|
| SUN RGB-D | yes | **yes**: the only suitable dataset |
| NYU Depth V2 (labeled) | yes | no: it has no 3D box annotations |

## Contents

- [Download the data](#download-the-data)
- [Inputs and outputs](#inputs-and-outputs)
- [Train / validation / test split](#train--validation--test-split)
- [Processed data](#processed-data)
- [Evaluation report](#evaluation-report)

## Download the data

```
pip install -r data/requirements.txt
python data/scripts/collect_data.py
```

This one command downloads both datasets (about 9.8 GiB), verifies size and SHA-256, extracts them and writes the training-ready files to `data/processed/` (about 3.5 GiB). The sources and checksums are in `manifests/datasets.json`; the script is `scripts/collect_data.py`. Details and options are in [Usage](#usage).

## Inputs and outputs

| | What | Where |
|---|---|---|
| **Input** | RGB image (`rgb`), depth map in mm (`depth`; `depth_raw` is the unfilled sensor depth) | `processed/<dataset>/<split>/{rgb,depth,depth_raw}/` |
| **Output (segmentation)** | semantic mask (`semantic`, one class per pixel) and instance mask (`instance`, one id per object) | `processed/<dataset>/<split>/{semantic,instance}/` |
| **Output (3D detection)** | 3D bounding boxes of the objects. **SUN RGB-D only**; NYU Depth V2 has none, so it is not used for detection | `processed/sunrgbd/boxes3d.jsonl` (see [3D boxes](#3d-boxes-sun-rgb-d-only)) |

Every sample has the same inputs and segmentation outputs in the `train`, `val` and `test` sets.

## Setup

```
pip install -r data/requirements.txt
```

`scipy` is pinned below 1.18 and `h5py` below 3.16, because Windows Smart App Control blocks scipy 1.18.x and h5py 3.16.0.

## Usage

Run all commands from the repository root.

Download, verify, extract and prepare both datasets with one command:

```
python data/scripts/collect_data.py --datasets sunrgbd nyu_depth_v2
```

With no step flag (`--verify`, `--extract`, `--prepare`), every step runs; with flags, only the given steps run.

| Flag | Effect |
|---|---|
| `--datasets` | Datasets to process (default: `[collect].datasets` in `config.toml`) |
| `--verify` | Check size and SHA-256 of downloads, then the dataset structure |
| `--extract` | Extract archives into `raw/` |
| `--prepare` | Write `processed/<dataset>/` and `index.jsonl` (including the train/val split) |
| `--boxes3d` | Only (re)write `boxes3d.jsonl` of an already prepared SUN RGB-D (`--prepare` does it too) |
| `--resplit` | Only redo the train/val split in `index.jsonl`; fast, no image is touched |
| `--val-ratio` | Validation fraction of the official train set (default: `[split] val_ratio`) |
| `--seed` | Seed of the split (default: `[split] seed`) |
| `--verify-only` | Check downloads, dataset structure and index; writes nothing |
| `--force` | Download and extract again |
| `--include-raw-nyu` | Not supported: stops with an error (see [Raw NYU](#raw-nyu)) |

Behaviour:

- **Idempotent**: a second run downloads, extracts and converts nothing that is already there; it only fills in missing files.
- **Resumable**: an interrupted download continues from its `.part` file.
- **Atomic**: every output is written to a temporary file first, so no half-written files are left behind.
- **Errors**: a failed check prints `ERROR: ...` and exits with code 1.
- `raw/` is never modified; all conversions go to `processed/`.

### Raw NYU

Only the labeled NYU subset (`nyu_depth_v2_labeled.mat`, ~2.8 GB, 1449 samples) is used; the team does not need the full raw dataset (~428 GB). `--include-raw-nyu` (or `include_raw_nyu = true` in `config.toml`) therefore stops with an error instead of downloading. If it is ever needed, download it manually from the [official NYU Depth V2 page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) into a folder outside `data/`; it is not handled by the collector or the index.

Evaluate the prepared data (quick test on 50 samples; omit the flag for all samples):

```
python data/scripts/data_evaluation.py --sample-limit 50
```

All parameters are in `config.toml`, download sources (URL, size, SHA-256) in `manifests/datasets.json`.

### Run times

Measured on one machine, only as a rough guide:

| Step | Time |
|---|---|
| `--prepare` NYU, first run | ~1.5 min |
| `--prepare` SUN, first run | ~11 min |
| Full run when everything is already prepared | ~30 s |
| `data_evaluation.py`, all samples (structure, rgb, depth, labels, alignment) | ~13.5 min |

## Layout

| Folder | Content |
|---|---|
| `downloads/` | Downloaded archives (not in Git) |
| `raw/` | Extracted, unmodified official data (not in Git) |
| `processed/` | Training-ready data and `index.jsonl` (not in Git) |
| `reports/` | Run summary (`collect_summary.json`) and evaluation reports; only `data_evaluation.md` and `problematic_samples.jsonl` are in Git, the rest is regenerated by the scripts |
| `src/` | Library code (`DataCollector`, `DataEvaluator`) |
| `scripts/` | Command-line entry points; `inspect_mat.py` and `inspect_sun.py` are debugging helpers |

The collector needs `min_free_space_gb` (40 GB) of free disk space. Actual usage: downloads 9.8 GiB (SUN 7.0, NYU 2.8), processed 3.5 GiB (SUN 3.1, NYU 0.4).

## Train / validation / test split

The datasets only come with an official train/test split. The validation set is a random part of the **official train** set; the official test set is never changed.

- The ratio is `[split] val_ratio` in `config.toml` (default `0.1`, i.e. 10 % of train); `0` means no validation set.
- The choice is random but fixed by `[split] seed` (default `42`): the same ratio and seed always give the same validation set.
- Only the `split` field in `index.jsonl` changes. The validation files stay in `train/`, so nothing is moved or copied, and changing the ratio takes seconds.

Change the ratio on already prepared data, for example 20 % validation or a different seed:

```
python data/scripts/collect_data.py --resplit --val-ratio 0.2
python data/scripts/collect_data.py --resplit --val-ratio 0.2 --seed 7
```

or set `val_ratio` in `config.toml` and run `--prepare` or `--resplit`. To use the sets, read `index.jsonl` and filter on `split` (see the example below).

SUN RGB-D and NYU Depth V2 contain many similar frames from the same scene, so a random split can put near-identical images into train and validation. If this matters for model selection, check it with `data_evaluation.py` (near-duplicate section).

## Processed data

```
processed/<dataset>/
  index.jsonl
  classes.json
  train/{rgb,depth,depth_raw,semantic,instance}/<id>.<ext>   (also holds the val samples)
  test/...
```

| Dataset | Samples | Official train | Train | Val | Test | Resolution (W×H) |
|---|---|---|---|---|---|---|
| `sunrgbd` | 10335 | 5285 | 4757 | 528 | 5050 | per sensor: kv2 730×530, xtion 591×441, kv1 561×427, realsense 681×531 |
| `nyu_depth_v2` | 1449 | 795 | 715 | 80 | 654 | 640×480 |

Train and val are the default `val_ratio = 0.1` of the official train set. The test sets are the official ones (`allsplit.mat` for SUN, `splits.mat` for NYU). File names are the original sample numbers (1-based), so e.g. NYU `train/` starts at `000003`.

```python
import json
rows = [json.loads(line) for line in open("data/processed/nyu_depth_v2/index.jsonl")]
val = [r for r in rows if r["split"] == "val"]  # paths in r are relative to processed/<dataset>/
```

### Index fields

One JSON object per line. All paths are relative to `processed/<dataset>/`.

```
{"id": "sun_000001", "dataset": "sunrgbd", "split": "test",
 "rgb": "test/rgb/000001.jpg", "depth": "test/depth/000001.png",
 "depth_raw": "test/depth_raw/000001.png", "semantic": "test/semantic/000001.png",
 "instance": "test/instance/000001.png", "scene": "furniture_store",
 "sensor": "kinect_v2", "source": "SUNRGBD/kv2/kinect2data/..."}
```

| Field | Meaning |
|---|---|
| `id` | `sun_NNNNNN` or `nyu_NNNNNN` |
| `split` | `train`, `val` or `test`; `val` rows point to files in `train/` |
| `rgb`, `depth`, `semantic`, `instance` | Image paths |
| `depth_raw` | Raw sensor depth (holes = 0) |
| `scene` | Scene type (`unknown` if missing) |
| `sensor` | `kinect_v1`, `kinect_v2`, `realsense` or `xtion` |
| `source` | SUN only: original sample folder under `raw/sunrgbd/` |

### 3D boxes (SUN RGB-D only)

`processed/sunrgbd/boxes3d.jsonl` has one line per sample (same `id` as in `index.jsonl`, all splits): 64657 boxes in 10335 samples, and 44 samples have none. NYU Depth V2 has no 3D annotation, so it has no such file.

```
{"id": "sun_000001",
 "K": [[529.5, 0, 365], [0, 529.5, 265], [0, 0, 1]],
 "Rtilt": [[...], [...], [...]],
 "boxes": [{"classname": "bed", "centroid": [1.05, 4.17, -0.25], "size": [1.89, 2.30, 1.97],
            "basis": [[...], [...], [...]], "orientation": [-0.12, -0.99, 0.0]}, ...]}
```

| Field | Meaning |
|---|---|
| `K` | Camera intrinsics (pixels) |
| `Rtilt` | Rotation from the camera points to the upright system |
| `centroid` | Box centre in metres |
| `size` | Full edge lengths in metres along the columns of `basis` (the source file stores half sizes) |
| `basis` | 3×3 rotation of the box (the box is rotated around the vertical axis only) |
| `orientation` | Horizontal direction of the box |
| `classname` | Object name (several hundred names; there is no fixed class list) |

The boxes are in the **upright coordinate system** of the SUN RGB-D toolbox (z up). A depth map `z` (metres, `depth` / 1000) gives points in this system like this (pixel indices `x`, `y` start at 1):

```python
pts = np.stack([(x - K[0, 2]) * z / K[0, 0], z, -(y - K[1, 2]) * z / K[1, 1]], -1)
pts = pts.reshape(-1, 3) @ Rtilt.T
inside = (np.abs((pts - centroid) @ basis.T) <= size / 2).all(axis=1)  # points of one box
```

Checked on 10 real samples (all four sensors): every one of the 36 boxes contained at least 200 depth points.

### File formats

| Field | Format | SUN RGB-D | NYU Depth V2 |
|---|---|---|---|
| `rgb` | JPEG | uint8, copied unchanged | uint8, quality 95 |
| `depth` | PNG | uint16, mm, hole-filled (`depth_bfx`) | uint16, mm, hole-filled (`depths`) |
| `depth_raw` | PNG | uint16, mm, 0 = no measurement | uint16, mm, 0 = no measurement (`rawDepths`) |
| `semantic` | PNG | **uint8**, 37 classes | **uint16**, 894 classes |
| `instance` | PNG | **uint16** | **uint8** |

`classes.json` is `["unlabeled", ...names]`: label 0 means unlabeled, label k is `classes[k]`. Note that the semantic and instance dtypes are swapped between the two datasets, and the class sets differ (37 vs 894), so joint training needs a common class mapping.

Depth values in both datasets are capped at 10 m (max 10000 mm) by the original data. The filled `depth` has no holes; how much of it was actually measured is shown by `depth_raw` (on average 84 % for SUN, 69 % for NYU).

## Evaluation report

`data_evaluation.py` reads the same `index.jsonl` as training and writes `reports/data_evaluation.json` and `reports/data_evaluation.md`. `--sample-limit N` checks N samples picked at random with a fixed seed, so repeated runs check the same samples; counts and missing-file checks always use the full index. It also writes `reports/problematic_samples.jsonl` (one flagged issue per line, with `severity` error or warning) and prints a short summary per dataset. The exit code is 1 if any sample has an error (missing, unreadable or malformed file, broken index) or an index is missing; warnings (dark images, little measured depth, misalignment, ...) only go to the report.

| Part | Status | Content |
|---|---|---|
| program | done | git commit, Python, numpy and Pillow versions, manifest SHA-256 |
| `structure` | done | name, version, source files and SHA-256, disk usage, processed file count, sample counts and train ratio, index duplicates, missing files and incomplete samples, format / dtype / resolution per field, unreadable files |
| `rgb` | done | per-channel mean, std, min, max and percentiles (0-1 scale, exact from histograms), image brightness, per-sensor means, dark / bright / low-contrast / broken images (flagged, not removed) |
| `depth` | done | unit and scale, min / max / mean / median / std and percentiles over valid pixels (exact from histograms), zero / negative / NaN / inf / invalid ratios, valid ratio per sample, implausible values (< 0.1 m, > 10 m), per-sensor means, raw vs. filled difference; samples whose raw depth is less than 50 % measured are flagged |
| `labels` | done | class count and names, pixel frequency per class, classes per image, instances, unlabeled / ignore ratio, 2D and 3D boxes (SUN), rare classes and imbalance; zero-size 3D boxes are flagged |
| `alignment` | done | scene and sensor counts per split, scenes in one split only, per-sample resolution match, RGB-depth edge alignment (best shift per sensor; samples > 2 px off and clearly better shifted are flagged), exact (SHA-256) and near (difference hash) duplicates within and across splits, SUN / NYU overlap, summary of samples to exclude |

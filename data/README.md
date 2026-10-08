# Data

Download, verification and preparation of the SUN RGB-D (V1) and NYU Depth V2 (labeled) datasets.

## Setup

```
pip install -r data/requirements.txt
```

`scipy` is pinned below 1.18, because Windows Smart App Control blocks 1.18.x.

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
| `--prepare` | Write `processed/<dataset>/` and `index.jsonl` |
| `--verify-only` | Check downloads, dataset structure and index; writes nothing |
| `--force` | Download and extract again |
| `--include-raw-nyu` | Full raw NYU (~428 GB); not supported yet, only logs a warning |

Behaviour:

- **Idempotent**: a second run downloads, extracts and converts nothing that is already there; it only fills in missing files.
- **Resumable**: an interrupted download continues from its `.part` file.
- **Atomic**: every output is written to a temporary file first, so no half-written files are left behind.
- **Errors**: a failed check prints `ERROR: ...` and exits with code 1.
- `raw/` is never modified; all conversions go to `processed/`.

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
| `reports/` | Run summary (`collect_summary.json`) and evaluation reports |
| `src/` | Library code (`DataCollector`, `DataEvaluator`) |
| `scripts/` | Command-line entry points; `inspect_mat.py` and `inspect_sun.py` are debugging helpers |

The collector needs `min_free_space_gb` (40 GB) of free disk space. Actual usage: downloads 9.8 GiB (SUN 7.0, NYU 2.8), processed 3.5 GiB (SUN 3.1, NYU 0.4).

## Processed data

```
processed/<dataset>/
  index.jsonl
  classes.json
  train/{rgb,depth,depth_raw,semantic,instance}/<id>.<ext>
  test/...
```

| Dataset | Samples | Train | Test | Resolution (W×H) |
|---|---|---|---|---|
| `sunrgbd` | 10335 | 5285 | 5050 | per sensor: kv2 730×530, xtion 591×441, kv1 561×427, realsense 681×531 |
| `nyu_depth_v2` | 1449 | 795 | 654 | 640×480 |

Splits are the official ones (`allsplit.mat` for SUN, `splits.mat` for NYU). File names are the original sample numbers (1-based), so e.g. NYU `train/` starts at `000003`.

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
| `split` | `train` or `test` |
| `rgb`, `depth`, `semantic`, `instance` | Image paths |
| `depth_raw` | Raw sensor depth (holes = 0) |
| `scene` | Scene type (`unknown` if missing) |
| `sensor` | `kinect_v1`, `kinect_v2`, `realsense` or `xtion` |
| `source` | SUN only: original sample folder under `raw/sunrgbd/` |

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

`data_evaluation.py` reads the same `index.jsonl` as training and writes `reports/data_evaluation.json` and `reports/data_evaluation.md`. `--sample-limit N` checks N samples picked at random with a fixed seed, so repeated runs check the same samples; counts and missing-file checks always use the full index.

| Part | Status | Content |
|---|---|---|
| program | done | git commit, Python, numpy and Pillow versions |
| `structure` | done | name, version, source files and SHA-256, disk usage, sample counts and train ratio, index duplicates, missing files and incomplete samples, format / dtype / resolution per field, unreadable files |
| `rgb` | done | per-channel mean, std, min, max and percentiles (0-1 scale, exact from histograms), image brightness, per-sensor means, dark / bright / low-contrast / broken images (flagged, not removed) |
| `depth` | done | unit and scale, min / max / mean / median / std and percentiles over valid pixels (exact from histograms), zero / negative / NaN / inf / invalid ratios, valid ratio per sample, implausible values (< 0.1 m, > 10 m), per-sensor means, raw vs. filled difference; samples whose raw depth is less than 50 % measured are flagged |
| `labels` | done | class count and names, pixel frequency per class, classes per image, instances, unlabeled / ignore ratio, 2D and 3D boxes (SUN), rare classes and imbalance; zero-size 3D boxes are flagged |
| `alignment` | done | scene and sensor counts per split, scenes in one split only, per-sample resolution match, RGB-depth edge alignment (best shift per sensor; samples > 2 px off and clearly better shifted are flagged), exact (SHA-256) and near (difference hash) duplicates within and across splits, SUN / NYU overlap, summary of samples to exclude |

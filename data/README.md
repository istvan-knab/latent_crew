# Data

Download, verification and preparation of the SUN RGB-D and NYU Depth V2 datasets.

## Setup

```
pip install -r data/requirements.txt
```

## Usage

Download, verify and extract both datasets:

```
python data/scripts/collect_data.py --datasets sunrgbd nyu_depth_v2 --verify --extract --prepare
```

Verify existing files only:

```
python data/scripts/collect_data.py --datasets sunrgbd nyu_depth_v2 --verify-only
```

Evaluate the prepared data (quick test on 100 samples):

```
python data/scripts/data_evaluation.py --sample-limit 100
```

All parameters are in `config.toml`, download sources in `manifests/datasets.json`.

## Layout

| Folder | Content |
|---|---|
| `downloads/` | Downloaded archives (not in Git) |
| `raw/` | Extracted, unmodified official data (not in Git) |
| `processed/` | Training-ready data and `index.jsonl` (not in Git) |
| `reports/` | Run summaries and evaluation reports |
| `src/` | Library code (`DataCollector`, `DataEvaluator`) |
| `scripts/` | Command-line entry points |

Needs about 30 GB of free disk space.
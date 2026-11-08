# Latent Crew

![Python](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)

**Team name:** Latent Crew

## Team

| | | | |
| --- | --- | --- | --- |
| **Name** | Istvan Gellert Knab | Botond Veress | Agoston Horvath |
| **Neptun code** | BWKUGQ | H2J0GS | GI41M1 |

## Project

Latent Crew builds 3D object detection and segmentation for indoor scenes on two RGB-D datasets, **SUN RGB-D** and **NYU Depth V2**. Only SUN RGB-D has 3D bounding box annotations, so it is the only one suitable for 3D object detection; NYU Depth V2 has semantic and instance masks only and is used for segmentation. The central idea is that one camera stream can be processed in several ways, so that different kinds of information are extracted from the same input: what objects are in the scene (semantic and instance segmentation) and where they are in 3D (3D bounding boxes).

Plan:

1. **Data**: download and prepare both datasets with train, validation and test inputs (RGB, depth) and outputs (segmentation masks for both datasets, 3D boxes for SUN RGB-D only).
2. **Models with transfer learning**: start from pretrained backbones to keep the computational cost low.
3. **Training and optimisation**: train the models, and improve them with [Weights & Biases](https://wandb.ai) sweeps and different hyperparameter optimisation methods.
4. **Simulation demo**: put the algorithms on a MicroDuck robot trained with a PPO policy in Isaac Sim. While the robot is steered through the built environment, it detects and segments the objects around it. If time runs out, the same is shown on a few pre-recorded videos.

## Milestones

| Milestone | Content | Where | Status |
| --- | --- | --- | --- |
| 1 | Data sources and download scripts, data exploration and visualisation, train/validation/test data preparation | Notebook: [`docs/milestones/milestone_1.ipynb`](docs/milestones/milestone_1.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/istvan-knab/latent_crew/blob/main/docs/milestones/milestone_1.ipynb)<br>Data code and description: [`data/`](data/README.md) | Done |
| 2 | - | - | Coming soon |

## Repo structure

```
data/       Dataset download, preparation and evaluation; data/README.md describes the data
docs/       Documentation; docs/milestones/ holds one notebook per milestone
docker/     Dockerfile, docker-compose.yml
scripts/    entry-point shell scripts (train/test per task, full app)
configs/    YAML config files
src/        latent_crew Python package (data, models, training, evaluation, utils)
app/        web app
temp/       2D MNIST baseline classifier (TensorFlow, W&B) used to try out the training workflow
```

Main files:

| File | Function |
| --- | --- |
| `docs/milestones/milestone_1.ipynb` | Milestone 1 notebook: documents both datasets (source, labels, quality, analysis) with plots; has an "Open in Colab" button |
| `data/scripts/collect_data.py` | Downloads, verifies, extracts and prepares the datasets; writes the train/val/test split (`index.jsonl`) and the SUN RGB-D 3D boxes (`boxes3d.jsonl`) |
| `data/scripts/data_evaluation.py` | Checks the prepared data (files, formats, labels, alignment, duplicates) and writes `data/reports/` |
| `data/src/` | Library code: `DataCollector`, `DataEvaluator`, config, file readers |
| `data/config.toml` | Paths, datasets, train/val split ratio and seed |
| `data/manifests/datasets.json` | Download URLs, sizes and SHA-256 of the dataset files |
| `data/reports/data_evaluation.md` | Result of the last data evaluation |
| `pyproject.toml`, `uv.lock` | Python environment of the `latent_crew` package |
| `docker/Dockerfile`, `docker/docker-compose.yml` | Reference Docker environment |
| `configs/default.yaml`, `scripts/*.sh`, `src/latent_crew/` | Training and evaluation skeleton for the next milestones (not implemented yet) |

## Software environment

- Python 3.14 (pinned in `.python-version`)
- Package manager: [uv](https://docs.astral.sh/uv/)
- Dependencies: declared in `pyproject.toml`, exact versions locked in `uv.lock` (PyTorch, torchvision, Open3D, OpenCV, NumPy, SciPy)
- Reference environment: the Docker image built from `docker/Dockerfile` (recommended for reproducibility)
- TODO: confirm the CUDA/cuDNN version used for training

## 🚀 How to run

### Milestone 1

Milestone 1 is the notebook **`docs/milestones/milestone_1.ipynb`**. Open it in Colab with the button in the [Milestones](#milestones) table, or locally:

```bash
pip install -r data/requirements.txt matplotlib ipykernel
python data/scripts/collect_data.py        # download and prepare the data (about 10 GB, 40 GB free space)
python data/scripts/data_evaluation.py     # data checks, writes data/reports/
jupyter notebook docs/milestones/milestone_1.ipynb
```

The notebook shows the stored results of the data pipeline; the data description and options (train/val/test ratio, 3D boxes) are in [`data/README.md`](data/README.md). On Colab the first cell can run the pipeline (`RUN_PIPELINE = True`).

### Later milestones (training, testing, app)

The commands below belong to the next milestones; the training and evaluation code is not implemented yet.

Every command can be run locally (with uv) or inside Docker. The same `latent-crew` Compose service runs every script.

### Setup

Local (uv):

```bash
uv sync
```

Docker:

```bash
docker compose -f docker/docker-compose.yml build
```

### Configuration

Set the dataset, model and hyperparameters in a config file (see `configs/default.yaml`):

```yaml
dataset:
  name: sunrgbd # or nyu_depth_v2
  root: data/sunrgbd
  num_classes: 37 # TODO: confirm per dataset

model:
  name: TODO # backbone/architecture not chosen yet
  task: detection # or segmentation

training:
  batch_size: TODO
  epochs: TODO
  lr: TODO
  optimizer: TODO
  weight_decay: TODO
  seed: TODO

hardware:
  device: cuda # TODO: confirm GPU/CUDA version used
  num_workers: TODO
```

To reproduce results on both datasets, repeat training/evaluation with `dataset.name` set to `sunrgbd` and `nyu_depth_v2`.

### Train detection

The training scripts check whether the datasets are present in `data/` and download them if they are missing (TODO: download not implemented yet). Checkpoints are written to `checkpoints/`.

```bash
scripts/train_detection.sh configs/default.yaml
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/train_detection.sh configs/default.yaml
```

### Train segmentation

```bash
scripts/train_segmentation.sh configs/default.yaml
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/train_segmentation.sh configs/default.yaml
```

### Test detection

```bash
scripts/test_detection.sh configs/default.yaml checkpoints/detection.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/test_detection.sh configs/default.yaml checkpoints/detection.pt
```

### Test segmentation

```bash
scripts/test_segmentation.sh configs/default.yaml checkpoints/segmentation.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/test_segmentation.sh configs/default.yaml checkpoints/segmentation.pt
```

### Run the app (detection + segmentation end-to-end)

```bash
scripts/run_app.sh configs/default.yaml checkpoints/detection.pt checkpoints/segmentation.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/run_app.sh configs/default.yaml checkpoints/detection.pt checkpoints/segmentation.pt
```

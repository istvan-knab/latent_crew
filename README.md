# Latent Crew

![Python](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)

3D object detection and segmentation on the SUN RGB-D and NYU Depth V2 datasets.

## Team

| Name | Neptun code |
| --- | --- |
| Istvan Gellert Knab | BWKUGQ |
| Botond Veress | H2J0GS |
| Agoston Horvath | GI41M1 |

## Repo structure

```
docker/     Dockerfile, docker-compose.yml
scripts/    entry-point shell scripts (train/test per task, full app)
configs/    YAML config files
src/        latent_crew Python package (data, models, training, evaluation, utils)
```

## Software environment

- Python 3.14 (pinned in `.python-version`)
- Package manager: [uv](https://docs.astral.sh/uv/)
- Dependencies: declared in `pyproject.toml`, exact versions locked in `uv.lock` (PyTorch, torchvision, Open3D, OpenCV, NumPy, SciPy)
- Reference environment: the Docker image built from `docker/Dockerfile` (recommended for reproducibility)
- TODO: confirm the CUDA/cuDNN version used for training

## 🚀 How to run

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

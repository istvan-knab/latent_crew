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

### Running on an NVIDIA GPU (DGX Spark)

The base Compose file has no GPU access, so it also runs on machines without an NVIDIA GPU (e.g. macOS). On a GPU machine add `docker/docker-compose.gpu.yml` (needs the NVIDIA Container Toolkit on the host, which DGX OS ships). Without it the container does not see the GPU, and with `hardware.device: cuda` training stops with `CUDA is not available` instead of silently training on the CPU.

**1. GPU check (run this first).** The locked `torch` has a Linux aarch64 wheel built against CUDA 13, but whether it contains kernels for the Spark's GB10 GPU (Blackwell, compute capability 12.1) is not verified yet. Expected: `True`, the GB10 device name and `(12, 1)`, without a warning about unsupported compute capability:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml run --rm latent-crew python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

If it prints `False`, a CUDA error or an unsupported-capability warning, install torch from the PyTorch CUDA index instead (configure it in `pyproject.toml` via `[tool.uv.index]` / `[tool.uv.sources]`, limited to Linux so macOS still resolves).

**2. Before the first training run:**

- **Weights & Biases:** log in once with `uv run wandb login`. For Docker, export the key in the host shell (`export WANDB_API_KEY=...`); Compose passes it into the container. Never put the key in the repo. Without W&B: `logging.mode: disabled` (or `offline`, sync later with `wandb sync`).
- **Data:** the first run downloads SUN RGB-D (~7 GB) and prepares it (~11 min); at least 40 GB of free disk space is needed (the collector checks it). Later runs reuse `data/processed/` and download nothing.
- **Working directory:** run every command from the repository root; `dataset.root` in the config is relative to it.
- **Pretrained weights:** the ImageNet encoder weights are downloaded on first use. Locally they are cached; in Docker there is no cache volume yet, so every container run downloads them again (slower start, not an error).

**3. Train:**

```bash
scripts/train_segmentation.sh configs/segmentation.yaml
```

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml run --rm latent-crew scripts/train_segmentation.sh configs/segmentation.yaml
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

Segmentation uses its own config, `configs/segmentation.yaml`: task label (`experiment.task: 2d_segmentation`), dataset, encoder, decoder, input modalities (RGB or RGB-D), hyperparameters and W&B logging. Options and best practices: [docs/segmentation_models.md](docs/segmentation_models.md).

At start the script checks `data/processed/<dataset>/index.jsonl` and the files it lists. If everything is there, it is used as is; only if something is missing does it run `data/scripts/collect_data.py` for that dataset (download, verify, extract, prepare; resumable, needs ~40 GB free space). Only `sunrgbd` can be trained for now (NYU needs the NYU40 class mapping).

Log in to Weights & Biases once (`uv run wandb login`) or set `WANDB_API_KEY`; set `logging.mode: disabled` to train without W&B. Every run is tagged with its task label.

```bash
scripts/train_segmentation.sh configs/segmentation.yaml
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/train_segmentation.sh configs/segmentation.yaml
```

Checkpoints go to `checkpoints/<experiment.name>/` (`best.pt` by validation mIoU, `last.pt`, `config.yaml`). Validation uses `training.val_fraction` of the official train split; the test split is only used by the test script.

Show one sample as the model sees it (RGB, depth, label mask, overlay; `--augment` applies the training augmentation):

```bash
scripts/visualize_segmentation.sh configs/segmentation.yaml --index 0
```

### Test detection

```bash
scripts/test_detection.sh configs/default.yaml checkpoints/detection.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/test_detection.sh configs/default.yaml checkpoints/detection.pt
```

### Test segmentation

Evaluates on the test split (mIoU, per-class IoU, pixel accuracy), prints the table, writes `<checkpoint>_test_metrics.json` and logs a W&B run tagged `2d_segmentation` and `evaluation`. The network is rebuilt from the model spec saved in the checkpoint.

```bash
scripts/test_segmentation.sh configs/segmentation.yaml checkpoints/sunrgbd_rgbd_resnet50_deeplabv3plus/best.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/test_segmentation.sh configs/segmentation.yaml checkpoints/sunrgbd_rgbd_resnet50_deeplabv3plus/best.pt
```

Tests: `uv run pytest -q` (includes an end-to-end training run on synthetic data, CPU, W&B off).

### Run the app (detection + segmentation end-to-end)

```bash
scripts/run_app.sh configs/default.yaml checkpoints/detection.pt checkpoints/segmentation.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/run_app.sh configs/default.yaml checkpoints/detection.pt checkpoints/segmentation.pt
```

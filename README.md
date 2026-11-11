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

- **Weights & Biases:** copy `.env.example` to `.env` in the repository root and fill in `WANDB_API_KEY` (`.env` is gitignored, never commit it). Docker Compose passes it into the container (`env_file`), and the train/test/HPO scripts load it for local runs too. Without it, the container asks for the key interactively and forgets it when it is removed. Without W&B: `logging.mode: disabled` (or `offline`, sync later with `wandb sync`).
- **Data:** the first run downloads SUN RGB-D (~7 GB) and prepares it (~11 min); at least 40 GB of free disk space is needed (the collector checks it). Later runs reuse `data/processed/` and download nothing.
- **Working directory:** run every command from the repository root; `dataset.root` in the config is relative to it.
- **Shared memory:** Compose gives the container `shm_size: 8gb`, because DataLoader workers pass batches through `/dev/shm` and Docker's 64 MB default fails with `unable to allocate shared memory(shm) ... No space left on device`. With plain `docker run`, add `--shm-size=8g`.
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

Put your Weights & Biases key in `.env` (`cp .env.example .env`, then fill in `WANDB_API_KEY`), or log in once locally with `uv run wandb login`; set `logging.mode: disabled` to train without W&B. Every run is tagged with its task label.

```bash
scripts/train_segmentation.sh configs/segmentation.yaml
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/train_segmentation.sh configs/segmentation.yaml
```

Checkpoints go to `checkpoints/<experiment.name>/` (`best.pt` by validation mIoU, `last.pt`, `config.yaml`). Validation uses `training.val_fraction` of the official train split; the test split is only used by the test script.

After successful training, the run logs a W&B artifact of type `model`, named `segmentation-<run_id>`, containing `best.pt`, `last.pt` and `config.yaml`. Find it in the run's **Artifacts** tab; its metadata includes the model specification, best validation mIoU and best epoch. Both checkpoints include the weights, model specification, class names and configuration needed to rebuild the network. With `logging.mode: online` the artifact is uploaded; `offline` stages it for later `wandb sync`, and `disabled` keeps only the local checkpoints. Existing checkpoints are not uploaded retroactively.

Docker keeps W&B artifact staging and its cache under the mounted `outputs/` directory, so they survive container removal. Sync offline Docker runs from the same Compose environment so their stored file paths resolve.

#### Hyperparameter optimisation

`configs/hpo_segmentation.yaml` holds the search space (with priors), the fidelity (training epochs per trial) and the total budget in epochs; everything else comes from the base config. Two algorithms are available through [NePS](https://github.com/automl/neps), chosen with `--optimizer`:

- `ifbo` (default): In-Context Freeze-Thaw BO, Rakotoarison et al., ICML 2024. Trains the configurations one epoch at a time and uses a pretrained learning-curve model to decide which one to continue. The surrogate (~55 MB) is downloaded on first use into `outputs/hpo/ifbo_surrogate/`.
- `priorband`: PriorBand, Mallik et al., NeurIPS 2023. Hyperband that samples around the priors (the base config) and the best configuration found so far.

```bash
scripts/hpo_segmentation.sh configs/segmentation.yaml configs/hpo_segmentation.yaml
```

```bash
scripts/hpo_segmentation.sh configs/segmentation.yaml configs/hpo_segmentation.yaml --optimizer priorband
```

Results go to `outputs/hpo/<study_name>/<optimizer>/`: one directory per trial step (config, report, and `state.pt` for the newest step of each configuration), NePS summaries in `summary/`, and `best_config.yaml`, which you can train directly with `scripts/train_segmentation.sh`. Running the same command again continues the study. In W&B:

- **Configuration runs:** one run per configuration, resumed at every step, grouped as `<study_name>-<optimizer>` and tagged `hpo`. Like plain training, every validation logs `logging.num_sample_images` sample predictions (set it to 0 to save space during long studies). With `logging.mode: online` they also join a W&B sweep of the same name, so the **Sweeps** page shows parallel coordinates and parameter importance. The sweep is display only: NePS picks the configurations, and W&B's own search never runs.
- **Study run:** `<study_name>-<optimizer>-study` (job type `hpo_study`) shows how the search went. It plots the best val mIoU over the spent epochs, which is the curve to compare optimizers by, and logs each step's decision (new configuration or continuation). It also has the learning curves of all configurations and the `study/trials` table, with one row per step and its hyperparameters. `best_config.yaml` is attached as an artifact at the end.

Details: [docs/segmentation_models.md](docs/segmentation_models.md#hyperparameter-optimisation).

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

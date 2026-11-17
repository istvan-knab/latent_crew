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
docker/     Dockerfile, docker-compose.yml (+ docker-compose.gpu.yml for NVIDIA GPUs)
scripts/    entry-point shell scripts (train, test, visualise)
configs/    YAML config files (configs/segmentation.yaml)
src/        latent_crew Python package (config, data, models, training, evaluation, utils)
tests/      pytest suite (runs on synthetic data, CPU only)
demo/       demo scripts (live RealSense view) and, later, the best models; see demo/README.md
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
| `configs/segmentation.yaml`, `scripts/*.sh`, `src/latent_crew/` | 2D semantic segmentation on SUN RGB-D: training, evaluation, visualisation |

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

### Segmentation (training and testing)

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

- **Weights & Biases:** copy `.env.example` to `.env` in the repository root and fill in `WANDB_API_KEY` (`.env` is gitignored, never commit it). Docker Compose passes it into the container (`env_file`), and the train/test scripts load it for local runs too. Without it, the container asks for the key interactively and forgets it when it is removed. Without W&B: `logging.mode: disabled` (or `offline`, sync later with `wandb sync`).
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

### Train segmentation

Segmentation uses its own config, `configs/segmentation.yaml`: dataset, encoder, decoder, input modalities (RGB or RGB-D), hyperparameters and W&B logging. Options and best practices: [docs/segmentation_models.md](docs/segmentation_models.md).

At start the script checks `data/processed/<dataset>/index.jsonl` and the files it lists. If everything is there, it is used as is; only if something is missing does it run `data/scripts/collect_data.py` for that dataset (download, verify, extract, prepare; resumable, needs ~40 GB free space). Only `sunrgbd` is supported.

Put your Weights & Biases key in `.env` (`cp .env.example .env`, then fill in `WANDB_API_KEY`), or log in once locally with `uv run wandb login`; set `logging.mode: disabled` to train without W&B.

```bash
scripts/train_segmentation.sh configs/segmentation.yaml
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/train_segmentation.sh configs/segmentation.yaml
```

The W&B dashboard of a training run has fixed sections (`src/latent_crew/utils/wandb_sections.py`):

| Section | Content |
|---|---|
| `loss/` | mean training loss per epoch and validation loss, and both together on one chart (`loss/train_vs_val`) |
| `metrics/` | validation mIoU, pixel accuracy, mean class accuracy, per-class IoU bar chart, learning rate at the end of each epoch |
| `samples/` | the same `logging.num_sample_images` validation images every epoch: RGB, depth, ground truth, prediction, ground-truth overlay, prediction overlay |
| `optim/` | per-iteration training loss and learning rate per parameter group |

Early stopping: `training.early_stopping_patience: 10` (in `configs/segmentation.yaml`) stops after 10 validations without a val mIoU gain; 0 turns it off. The best checkpoint is kept either way.

Checkpoints go to `checkpoints/<experiment.name>/` (`best.pt` by validation mIoU, `last.pt`, `config.yaml`). Validation uses the `val` split of `index.jsonl` (SUN RGB-D: 528 images, a fixed seeded 10 % of the official train set, set by `[split] val_ratio` in `data/config.toml`); training uses the remaining 4757 `train` images. The test split is only used by the test script.

After successful training, the run logs a W&B artifact of type `model`, named `segmentation-<run_id>`, containing `best.pt`, `last.pt` and `config.yaml`. Find it in the run's **Artifacts** tab; its metadata includes the model specification, best validation mIoU and best epoch. Both checkpoints include the weights, model specification, class names and configuration needed to rebuild the network. With `logging.mode: online` the artifact is uploaded; `offline` stages it for later `wandb sync`, and `disabled` keeps only the local checkpoints. Existing checkpoints are not uploaded retroactively.

Docker keeps W&B artifact staging and its cache under the mounted `outputs/` directory, so they survive container removal. Sync offline Docker runs from the same Compose environment so their stored file paths resolve.

Show one sample as the model sees it (RGB, depth, label mask, overlay; `--augment` applies the training augmentation):

```bash
scripts/visualize_segmentation.sh configs/segmentation.yaml --index 0
```

### Test segmentation

Evaluates on the test split (mIoU, per-class IoU, pixel accuracy), prints the table, writes `<checkpoint>_test_metrics.json` and logs a W&B run tagged `evaluation`. The network is rebuilt from the model spec saved in the checkpoint.

```bash
scripts/test_segmentation.sh configs/segmentation.yaml checkpoints/sunrgbd_rgbd_resnet50_deeplabv3plus/best.pt
```

```bash
docker compose -f docker/docker-compose.yml run --rm latent-crew scripts/test_segmentation.sh configs/segmentation.yaml checkpoints/sunrgbd_rgbd_resnet50_deeplabv3plus/best.pt
```

Tests: `uv run pytest -q` (includes an end-to-end training run on synthetic data, CPU, W&B off).

### Demo: live RealSense camera view

The `demo/` folder holds everything for the demo (see [`demo/README.md`](demo/README.md)). `demo/realsense_stream.py` shows the RGB and the colourised depth stream of a plugged-in Intel RealSense camera (e.g. D435i) side by side, with FPS; `q` or Esc quits. It runs locally with uv (no Docker):

```bash
demo/realsense_stream.sh
```

- **SDK:** comes from the optional `camera` extra in `pyproject.toml` (`pyrealsense2-macosx` on macOS, `pyrealsense2` on Linux/Windows); plain `uv sync` and Docker leave it out.
- **Supported:** macOS 15+ on Apple Silicon, Linux x86_64 and Windows, Python 3.14. Linux aarch64 (e.g. the DGX Spark) has no wheel; build librealsense from source there.
- **macOS USB access:** if the camera is not found or access is refused, run `demo/realsense_stream.sh --sudo`. It installs the environment as your user and only runs the viewer as root.
- **USB 3:** use a USB 3 capable USB-C port and cable, ideally without a hub; over USB 2 the depth stream fails to start or only runs at low resolution and frame rate.
- Options: `--width`, `--height`, `--fps`, `--max-depth` (metres at the far end of the colour scale).

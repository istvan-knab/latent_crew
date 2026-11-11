# Segmentation models

2D semantic segmentation on SUN RGB-D and NYU Depth V2: every pixel gets a class label. The network is an **encoder** (feature extractor, pretrained) followed by a **decoder** (restores the resolution and classifies each pixel). Both are chosen in [`configs/segmentation.yaml`](../configs/segmentation.yaml). Implemented: model factory, config, Dataset and augmentation (`src/latent_crew/data/`), training loop (`training/train_segmentation.py`) and test evaluation (`evaluation/test_segmentation.py`). Not wired in yet: auxiliary head, OHEM, test-time augmentation.


```yaml
model:
  decoder: deeplabv3plus
  encoder: resnet50
  encoder_weights: imagenet
dataset:
  modalities: [rgb, depth] # [rgb] -> 3 input channels, [rgb, depth] -> 4
```

Models are built with [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (smp 0.5.0) in [`src/latent_crew/models/segmentation.py`](../src/latent_crew/models/segmentation.py). Every decoder below was smoke-tested with CNN and transformer encoders and with 3 and 4 input channels (`tests/test_segmentation_models.py`).

**Status column:** *config* = selectable in the config today; *reference* = literature / future work, not wired in.

## Encoders

The encoder downsamples the image in stages (1/4, 1/8, 1/16, 1/32 resolution) and learns more and more abstract features.

| Family | `model.encoder` values | Strengths | Weaknesses | Status |
|---|---|---|---|---|
| Classic CNN | `resnet50`, `resnet101`, `resnext50_32x4d` | stable, well understood, most examples use it | not the strongest today | config |
| Modern CNN | `tu-convnext_tiny`, `tu-convnext_small`, `efficientnet-b3` | near-transformer accuracy, CNN simplicity | ConvNeXt uses LayerNorm: transformer-style recipe (AdamW) | config |
| Lightweight | `mobilenet_v2`, `efficientnet-b0` | fast, fits small GPUs, good for debugging | lower accuracy | config |
| Hierarchical transformer | `mit_b0` ... `mit_b5` (SegFormer encoders) | global context, strong for segmentation | sensitive to hyperparameters, needs AdamW and warmup | config |
| Swin transformer | `tu-swin_tiny_patch4_window7_224` | strong with `upernet` in the literature | the timm model is fixed to 224×224 input: fails with our 480×640 crop, and fails with `deeplabv3plus` | reference |
| Foundation model | DINOv2 (ViT) | very strong generic features, good with little data | single scale (patch 14), no feature pyramid, heavy | reference |
| RGB-D two-branch | ESANet, CMX, DFormer | separate RGB and depth branches fused per stage | more complex, no ready-made smp encoder | reference |

`tu-<name>` selects any [timm](https://github.com/huggingface/pytorch-image-models) model as encoder. Full smp list: `smp.encoders.get_encoder_names()`.

### Feeding depth to the encoder

| Fusion | How | When | Status |
|---|---|---|---|
| Early | depth is a 4th input channel, the first conv is widened | first RGB-D experiment, simplest | config (`modalities: [rgb, depth]`) |
| Middle | two encoder branches fused at every stage (ESANet, CMX) | gives the best results in the literature | reference |
| Late | two full networks, outputs merged | rarely worth it, twice the cost | reference |

How the 4th channel is initialised (checked in the installed source): smp's `patch_first_conv` and timm's `adapt_input_conv` both **copy the pretrained RGB filters cyclically** (channel 4 gets the R filter) and **scale all weights by 3/4**, so the activation magnitude matches the 3-channel network. It is not an average of the RGB filters.

Depth representations: raw depth normalised with dataset statistics (used by `SegmentationDataset`), HHA (horizontal disparity, height above ground, angle with gravity; classic, expensive to compute), or depth repeated to 3 channels for a second pretrained branch.

### Encoder best practices

1. **Always start from pretrained weights** (`encoder_weights: imagenet`). With 795 (NYU) to 5285 (SUN) training images this matters more than the architecture.
2. **Lower learning rate for the encoder** than for the decoder: `training.encoder_lr_mult` (default 0.1). For transformers, layer-wise LR decay is the stronger variant.
3. **Normalisation:** ImageNet mean/std for RGB; depth with its own statistics (SUN: mean ≈ 2398 mm, std ≈ 1249 mm, see `data/reports/data_evaluation.md`).
4. **BatchNorm with small batches:** below ~8 images per GPU BatchNorm statistics get noisy; freeze BN or use GroupNorm/SyncBN.
5. **Output stride:** DeepLab decoders replace the last strides with dilation (`decoder_kwargs: {encoder_output_stride: 16}`, or 8). Lower stride = sharper but more memory; 16 is the usual compromise.
6. **Input size** must be a multiple of 32 (`augmentation.crop_size` is validated).

## Decoders

The decoder trades off **context** (what is around a pixel) against **detail** (exact boundaries).

| `model.decoder` | Idea | Pairs well with | Status |
|---|---|---|---|
| `unet` | symmetric, skip connection at every level | any; sharp boundaries, small data | config |
| `fpn` | feature pyramid, lightweight | any | config |
| `deeplabv3` | ASPP: parallel dilated convs, multi-scale context | ResNet | config |
| `deeplabv3plus` | DeepLabV3 + one low-level skip for edges | ResNet, ConvNeXt; **default starting point** | config |
| `pspnet` | pyramid pooling, global context | ResNet | config |
| `upernet` | pyramid pooling + FPN | ConvNeXt, MiT (standard head for Swin/ConvNeXt) | config |
| `segformer` | all-MLP head, very light | MiT (`mit_b*`) | config |
| FCN | plain upsampling | baseline only | reference |
| Mask2Former | mask classification with queries | strongest, but complex and heavy | reference |

### Decoder best practices

1. **Match the decoder to the encoder:** CNN → `deeplabv3plus` / `fpn` / `unet`; ConvNeXt → `upernet`; MiT → `segformer`.
2. **Skip connections** from early, high-resolution stages are needed for sharp boundaries (chair legs, door frames).
3. **Upsample the logits** to full resolution (bilinear) before the loss; never downsample the label mask with anything but nearest.
4. **Auxiliary head (deep supervision):** an extra classifier on an intermediate encoder stage, ~0.4 loss weight, dropped at inference. Not wired in yet.
5. **Dropout** (~0.1) before the final classifier.
6. The decoder is **randomly initialised**, so it gets the full learning rate.

## How training works

Encoder and decoder are trained **together, end to end**, in one loop. The encoder is not trained first:

- the encoder starts from **pretrained** ImageNet weights (it already "sees" edges, textures, objects),
- the decoder and the classifier head start **random**,
- every step: forward pass → pixel-wise cross-entropy → backpropagation through decoder **and** encoder → one optimizer step.

Two optional refinements, both in `training`:

| Setting | What it does | Why |
|---|---|---|
| `encoder_lr_mult: 0.1` | encoder LR = 0.1 × decoder LR (two optimizer parameter groups) | pretrained weights only need small adjustments; the random decoder must move a lot |
| `freeze_encoder_epochs: N` | for the first N epochs only the decoder is trained, then the encoder is unfrozen | early decoder gradients are noisy and can damage the pretrained features; often 0 is fine |

Helpers: `param_groups()` and `set_encoder_trainable()` in `models/segmentation.py`.

## Shared training practices

- **Labels:** `SegmentationDataset` remaps label 0 (unlabeled, ~25 % of SUN pixels) to `ignore_index: 255` and class k to k−1, so the head has `num_classes` outputs (SUN 37, NYU 40). Ignored pixels are left out of the loss and the mIoU.
- **Loss:** `CrossEntropyLoss(ignore_index=255)`. Against class imbalance (wall 26.6 %, floor 23.7 % on SUN): `class_weighting: median_frequency` (implemented), or OHEM (not implemented).
- **Augmentation** (`src/latent_crew/data/transforms.py`): geometric ops (random scale 0.5–2.0, random crop, horizontal flip) identical for RGB, depth and mask, with nearest interpolation for mask and depth; colour jitter on RGB only.
- **Optimizer:** SGD (momentum 0.9) for CNN encoders, AdamW for transformer encoders.
- **Schedule:** poly or cosine with a few hundred warmup iterations.
- **Mixed precision** (`amp: true`): half the memory, faster.
- **Evaluation:** mIoU at full resolution, plus pixel accuracy. Test-time augmentation (multi-scale + flip) is reported separately.
- **Classes:** NYU40 mapping for NYU (not in `data/processed/` yet), 37 classes for SUN.
- **Checkpoints** store `SegmentationConfig.model_spec()` (decoder, encoder, in_channels, num_classes); `test_segmentation.py` rebuilds the network from it with `build_segmentation_model(spec, pretrained=False)`, so evaluation gets the exact network even if the config changes later.
- **W&B model artifacts:** after successful training, `segmentation-<run_id>` (type `model`) contains `best.pt`, `last.pt` and `config.yaml`, with the model specification and best validation result in its metadata. Each run has its own artifact name. Online runs upload it before finishing; offline runs stage it for `wandb sync`; disabled logging creates no artifact. Local checkpoints are retained in every mode.

## Hyperparameter optimisation

`training/hpo_segmentation.py` tunes the `training.*` hyperparameters with [NePS](https://github.com/automl/neps). Both algorithms are **multi-fidelity**: the fidelity is the number of training epochs, so weak configurations are dropped after a few epochs and the budget goes to the promising ones.

| `--optimizer` | Paper | How it spends the budget | Good when |
|---|---|---|---|
| `ifbo` (default) | Rakotoarison et al., *In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization*, ICML 2024 | trains one epoch at a time; a transformer pretrained on learning curves (FT-PFN) predicts how each curve continues and picks which configuration gets the next epoch, or starts a new one | early epochs already say something about the final mIoU (pretrained encoder) |
| `priorband` | Mallik et al., *PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning*, NeurIPS 2023 | Hyperband brackets (rungs ≈ 1, 3, 10, 30 epochs with `eta: 3`); new configurations come from the uniform distribution, the prior or around the incumbent | the base config is already a good guess (the priors) |

Survey for background: Bischl et al., *Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges*, WIREs DMKD 2023.

How it works:

- **Search space:** `search_space` in `configs/hpo_segmentation.yaml`; keys are `training.<field>` (float, int with optional `log`, or categorical). The defaults search LR, encoder LR multiplier, weight decay, momentum, warmup, schedule and class weighting. The priors (`prior`, `confidence`) match `configs/segmentation.yaml`, so change them together. PriorBand needs them; ifBO uses them for its initial design only when `ifbo.use_priors: true` (false is the paper's plain Sobol design). The optimizer type is not searched by default, because SGD and AdamW need LRs that differ by orders of magnitude.
- **Steps resume:** a trial step continues from the previous step's `state.pt` (weights, optimizer, LR scheduler, W&B run id), trains only the missing epochs, and deletes the older state. The budget (`budget_epochs`) counts epochs actually trained.
- **Fixed schedule horizon:** warmup and poly/cosine decay always span `fidelity.max_epochs`, so a configuration stopped at epoch k has seen exactly the first k epochs of its full run. `best_config.yaml` is written with `epochs: max_epochs`; training it longer changes the schedule, so retune or treat it as a starting point.
- **Objective:** 1 − validation mIoU (ifBO needs a value in [0, 1]). Validation is the `val_fraction` hold-out of the train split; the test split is never used for tuning.
- **Same start for every configuration:** every configuration starts from the same seed (same decoder init, same data order), so the differences come from the hyperparameters.
- **W&B:** besides one run per configuration there is a study run (`StudyLogger`). It logs the incumbent trajectory (`study/best_val_mIoU` against `study/epochs_spent`), every step's decision (`study/continued`, `study/config`), the learning curves of all configurations and the `study/trials` table. In online mode the configuration runs also join a display-only sweep (`method: custom` with a local controller that is never started), which gives W&B's parallel-coordinates and parameter-importance panels. The sweep and study run ids are kept in `<output_dir>/<study_name>/<optimizer>_wandb.yaml`, so continuing the study continues both. Offline and disabled modes create no sweep.
- **Constraints:** `fidelity.min_epochs` must exceed `freeze_encoder_epochs`, otherwise the shortest trials would only have trained the decoder. Disk: one `state.pt` per configuration (weights + optimizer state, roughly 2× the model size).

## Suggested experiment order

| Step | Encoder | Decoder | Modalities | Question |
|---|---|---|---|---|
| 1 | `resnet50` | `deeplabv3plus` | `[rgb]` | does the full pipeline work (baseline)? |
| 2 | `resnet50` | `deeplabv3plus` | `[rgb, depth]` | how much does depth add? |
| 3 | `mit_b2` or `tu-convnext_tiny` | `segformer` / `upernet` | `[rgb, depth]` | does a modern encoder help? |
| 4 | two-branch (CMX/DFormer style) | – | RGB + depth | middle fusion (reference only) |

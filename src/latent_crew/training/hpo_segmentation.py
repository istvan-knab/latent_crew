"""Hyperparameter optimisation of the segmentation training with NePS: ifBO (default) or PriorBand.

- ifBO: Rakotoarison et al., "In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization", ICML 2024.
  A pretrained transformer (FT-PFN) extrapolates the learning curves and picks which configuration to train one step more.
- PriorBand: Mallik et al., "PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning", NeurIPS 2023.
  Hyperband whose sampling mixes uniform, prior (the current config) and incumbent-centred proposals.

Both are multi-fidelity with the number of epochs as fidelity. A configuration is trained in steps (ifBO: one epoch,
PriorBand: up to the next Hyperband rung); every step resumes from the checkpoint of the previous one. The LR schedule
always spans fidelity.max_epochs, so a configuration stopped at epoch k has seen the first k epochs of its full run.
The objective is 1 - val mIoU (ifBO needs it in [0, 1]); the test split is never touched.
"""

import argparse
import copy
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

import neps
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from latent_crew.config import SegmentationConfig, TrainingConfig
from latent_crew.models.segmentation import build_segmentation_model
from latent_crew.training.train_segmentation import (build_criterion, build_optimizer, build_scheduler, class_weight_for,
                                                     load_train_val, loader_args, train_loader_for, train_one_epoch, validate)
from latent_crew.utils.runtime import init_wandb, resolve_device, seed_everything

OPTIMIZERS = ("ifbo", "priorband")
PARAM_TYPES = ("float", "int", "categorical")
CONFIDENCE_LEVELS = ("low", "medium", "high")
# epochs is the fidelity; the others are bookkeeping, not hyperparameters
NOT_SEARCHABLE = ("epochs", "eval_every", "log_every", "val_fraction", "amp")
FIDELITY = "epochs"
STATE_FILE = "state.pt"
IFBO_MAX_PARAMS = 10  # the FT-PFN was pretrained on curves of up to 10 hyperparameters


@dataclass
class ParamSpec:
    type: str
    lower: float | None = None
    upper: float | None = None
    log: bool = False
    choices: List[Any] | None = None
    prior: Any = None
    confidence: str = "medium"


@dataclass
class HpoConfig:
    study_name: str
    output_dir: str
    budget_epochs: int
    min_epochs: int
    max_epochs: int
    search_space: Dict[str, ParamSpec]
    ifbo: Dict[str, Any] = field(default_factory=dict)
    priorband: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, path: str | Path, base: SegmentationConfig) -> "HpoConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        hpo = cls(
            study_name=raw["study_name"],
            output_dir=raw["output_dir"],
            budget_epochs=raw["budget_epochs"],
            min_epochs=raw["fidelity"]["min_epochs"],
            max_epochs=raw["fidelity"]["max_epochs"],
            search_space={name: ParamSpec(**spec) for name, spec in raw["search_space"].items()},
            ifbo=raw.get("ifbo") or {},
            priorband=raw.get("priorband") or {},
        )
        hpo.validate(base)
        return hpo

    def validate(self, base: SegmentationConfig) -> None:
        if not 1 <= self.min_epochs < self.max_epochs:
            raise ValueError("fidelity must satisfy 1 <= min_epochs < max_epochs")
        if self.budget_epochs < self.max_epochs:
            raise ValueError("budget_epochs must allow at least one full run (>= fidelity.max_epochs)")
        if not self.search_space:
            raise ValueError("search_space is empty")
        searchable = {f.name for f in fields(TrainingConfig)} - set(NOT_SEARCHABLE)
        for name, spec in self.search_space.items():
            section, _, key = name.partition(".")
            if section != "training" or key not in searchable:
                raise ValueError(f"search_space key {name!r} must be training.<field>, field one of {sorted(searchable)}")
            if spec.type not in PARAM_TYPES:
                raise ValueError(f"{name}: type must be one of {PARAM_TYPES}")
            if spec.confidence not in CONFIDENCE_LEVELS:
                raise ValueError(f"{name}: confidence must be one of {CONFIDENCE_LEVELS}")
            if spec.type == "categorical":
                if not spec.choices:
                    raise ValueError(f"{name}: categorical needs choices")
                if spec.prior is not None and spec.prior not in spec.choices:
                    raise ValueError(f"{name}: prior {spec.prior!r} is not one of the choices")
                continue
            if spec.lower is None or spec.upper is None or not spec.lower < spec.upper:
                raise ValueError(f"{name}: needs lower < upper")
            if spec.log and spec.lower <= 0:
                raise ValueError(f"{name}: log scale needs lower > 0")
            if spec.prior is not None and not spec.lower <= spec.prior <= spec.upper:
                raise ValueError(f"{name}: prior {spec.prior} is outside [{spec.lower}, {spec.upper}]")
        # Before the encoder is unfrozen the curve says little about the full run, so the lowest fidelity must be past it
        freeze = self.search_space.get("training.freeze_encoder_epochs")
        max_freeze = freeze.upper if freeze is not None else base.training.freeze_encoder_epochs
        if max_freeze >= self.min_epochs:
            raise ValueError(f"fidelity.min_epochs ({self.min_epochs}) must exceed freeze_encoder_epochs ({max_freeze})")


def to_neps_parameter(spec: ParamSpec, use_priors: bool):
    prior = {} if not use_priors or spec.prior is None else {"prior_confidence": spec.confidence}
    if spec.type == "categorical":
        if prior:
            prior["prior"] = spec.choices.index(spec.prior)  # NePS takes the index of the prior choice
        return neps.Categorical(choices=tuple(spec.choices), **prior)
    if prior:
        prior["prior"] = spec.prior
    if spec.type == "int":
        return neps.Integer(lower=int(spec.lower), upper=int(spec.upper), log=spec.log, **prior)
    return neps.Float(lower=float(spec.lower), upper=float(spec.upper), log=spec.log, **prior)


def pipeline_space(hpo: HpoConfig, use_priors: bool) -> neps.PipelineSpace:
    attrs = {name: to_neps_parameter(spec, use_priors) for name, spec in hpo.search_space.items()}
    attrs[FIDELITY] = neps.IntegerFidelity(lower=hpo.min_epochs, upper=hpo.max_epochs)
    return type("SegmentationSpace", (neps.PipelineSpace,), attrs)()


class SegmentationStudy:
    """Data, split and class weights shared by all trials of one study; evaluate() is the NePS objective."""

    def __init__(self, base: SegmentationConfig, hpo: HpoConfig, optimizer: str):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if optimizer == "ifbo" and len(hpo.search_space) > IFBO_MAX_PARAMS:
            raise ValueError(f"ifBO's FT-PFN surrogate handles at most {IFBO_MAX_PARAMS} hyperparameters, "
                             f"the search space has {len(hpo.search_space)}")
        self.base, self.hpo, self.optimizer = base, hpo, optimizer
        self.root = Path(hpo.output_dir) / hpo.study_name / optimizer
        self.device = resolve_device(base.hardware.device)
        self.train_base, self.val_base, self.train_idx, val_idx = load_train_val(base)
        self.train_set, self.val_set = Subset(self.train_base, self.train_idx), Subset(self.val_base, val_idx)
        self.val_loader = DataLoader(self.val_set, batch_size=1, **loader_args(base, self.device))
        self._train_loaders: Dict[int, tuple[DataLoader, torch.Generator]] = {}
        self._class_weight: torch.Tensor | None = None

    def trial_config(self, params: Dict[str, Any], config_id: str) -> SegmentationConfig:
        raw = copy.deepcopy(self.base.to_dict())
        for name, value in params.items():
            spec = self.hpo.search_space[name]
            # Plain Python types, so the saved config.yaml stays readable; NePS samples floats in float32
            # precision (0.0009999999), 6 significant digits keep exactly what it chose
            value = {"float": lambda v: float(f"{float(v):.6g}"), "int": int}.get(spec.type, lambda v: v)(value)
            section, _, key = name.partition(".")
            raw[section][key] = value
        raw["training"]["epochs"] = self.hpo.max_epochs
        raw["experiment"]["name"] = f"{self.hpo.study_name}-{self.optimizer}-{config_id}"
        config = SegmentationConfig(raw)
        if len(self.train_idx) < config.training.batch_size:
            raise ValueError(f"Only {len(self.train_idx)} training samples for batch_size={config.training.batch_size}")
        return config

    def train_loader(self, config: SegmentationConfig) -> tuple[DataLoader, torch.Generator]:
        batch_size = config.training.batch_size
        if batch_size not in self._train_loaders:
            generator = torch.Generator()
            self._train_loaders[batch_size] = (train_loader_for(self.train_set, config, self.device, generator), generator)
        return self._train_loaders[batch_size]

    def class_weight(self, config: SegmentationConfig) -> torch.Tensor | None:
        if config.training.class_weighting != "median_frequency":
            return None
        if self._class_weight is None:  # one pass over the masks per study, not per trial
            self._class_weight = class_weight_for(config, self.train_base, self.train_idx).to(self.device)
        return self._class_weight

    def evaluate(self, pipeline_directory: Path, previous_pipeline_directory: Path | None, pipeline_id: str,
                 **params: Any) -> Dict[str, Any]:
        """Trains one configuration from its previous step (if any) up to `epochs`, then validates it."""
        target_epoch = int(params.pop(FIDELITY))
        config_id = pipeline_id.split("_")[0]  # "7_3" (ifBO) and "7_rung_2" (PriorBand) are both configuration 7
        config = self.trial_config(params, config_id)
        t = config.training

        state = None
        if previous_pipeline_directory is not None:
            state = torch.load(Path(previous_pipeline_directory) / STATE_FILE, map_location=self.device, weights_only=True)
        start_epoch, step = (state["epoch"], state["step"]) if state else (0, 0)
        if target_epoch <= start_epoch:
            raise ValueError(f"Trial {pipeline_id}: target epoch {target_epoch} <= already trained {start_epoch}")
        # Same seed for every configuration at the start (same initial decoder); resumed steps get fresh randomness
        seed_everything(config.experiment.seed + start_epoch)

        model = build_segmentation_model(config.model_spec(), pretrained=state is None).to(self.device)
        optimizer = build_optimizer(model, config)
        loader, generator = self.train_loader(config)
        scheduler = build_scheduler(optimizer, config, len(loader))
        if state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        criterion = build_criterion(config, self.class_weight(config))

        run = init_wandb(config, config.experiment.name, extra_tags=["hpo", self.optimizer],
                         run_id=state["wandb_run_id"] if state else None,  # one W&B run per configuration
                         group=f"{self.hpo.study_name}-{self.optimizer}")
        try:
            for epoch in range(start_epoch + 1, target_epoch + 1):
                generator.manual_seed(config.experiment.seed + epoch)
                step = train_one_epoch(model, loader, optimizer, scheduler, criterion, config, self.device, epoch, step, run)
            metrics = validate(model, self.val_loader, self.val_set, self.val_base, config, self.device, target_epoch, step, run, 0)
            run.summary["hpo/epochs"] = target_epoch
            run_id = run.id
        finally:
            run.finish()

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": target_epoch, "step": step, "wandb_run_id": run_id, "model_spec": config.model_spec()},
                   Path(pipeline_directory) / STATE_FILE)
        if previous_pipeline_directory is not None:
            # Only the newest step of a configuration is ever continued; older states would just fill the disk
            (Path(previous_pipeline_directory) / STATE_FILE).unlink(missing_ok=True)
        return {
            "objective_to_minimize": 1.0 - metrics["mIoU"],
            "cost": target_epoch - start_epoch,  # epochs trained in this step; the budget is counted in these
            "info_dict": {"val_mIoU": metrics["mIoU"], "val_pixel_accuracy": metrics["pixel_accuracy"],
                          "epochs": target_epoch, "wandb_run_id": run_id, "max_epochs": t.epochs},
        }

    def optimizer_kwargs(self) -> Dict[str, Any]:
        if self.optimizer == "priorband":
            return dict(self.hpo.priorband)
        kwargs = dict(self.hpo.ifbo)
        # The FT-PFN surrogate (~55 MB) is downloaded once into this directory and shared by all studies
        surrogate_dir = Path(kwargs.pop("surrogate_dir", Path(self.hpo.output_dir) / "ifbo_surrogate"))
        surrogate_dir.mkdir(parents=True, exist_ok=True)
        kwargs["surrogate_path"] = str(surrogate_dir.resolve())
        return kwargs

    def uses_priors(self) -> bool:
        return self.optimizer == "priorband" or bool(self.hpo.ifbo.get("use_priors", False))

    def run(self) -> Path:
        """Runs (or continues) the study until budget_epochs are spent; returns the path of best_config.yaml."""
        # No mkdir here: NePS treats an existing root directory as a study to continue
        print(f"HPO {self.optimizer}: {len(self.train_set)} train / {len(self.val_set)} val samples, "
              f"{self.hpo.budget_epochs} epochs budget, fidelity {self.hpo.min_epochs}-{self.hpo.max_epochs} epochs, "
              f"results in {self.root}")
        neps.run(
            self.evaluate,
            pipeline_space=pipeline_space(self.hpo, self.uses_priors()),
            root_directory=self.root,
            total_cost_to_spend=self.hpo.budget_epochs,
            optimizer=(self.optimizer, self.optimizer_kwargs()),
        )
        return self.write_best_config()

    def best_trial(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """(config, report) of the successful trial with the lowest objective, read from NePS' trial directories."""
        best = None
        for report_path in sorted((self.root / "configs").glob("*/report.yaml")):
            report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
            if report.get("reported_as") != "success" or report.get("objective_to_minimize") is None:
                continue
            if best is None or report["objective_to_minimize"] < best[1]["objective_to_minimize"]:
                best = (yaml.safe_load((report_path.parent / "config.yaml").read_text(encoding="utf-8")), report)
        if best is None:
            raise RuntimeError(f"No successful trial in {self.root}")
        return best

    def write_best_config(self) -> Path:
        """Base config + best hyperparameters, with the epochs the schedule was tuned for; train it with train_segmentation."""
        params, report = self.best_trial()
        config = self.trial_config({k: v for k, v in params.items() if k in self.hpo.search_space}, "best")
        config.experiment.name = f"{self.hpo.study_name}_{self.optimizer}_best"
        path = self.root / "best_config.yaml"
        config.save_yaml(path)
        print(f"Best val mIoU {1.0 - report['objective_to_minimize']:.4f} after {params[FIDELITY]} epochs: "
              f"{ {k: params[k] for k in self.hpo.search_space} }\nConfig for the final training: {path}")
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation of the segmentation training (NePS).")
    parser.add_argument("--config", required=True, help="Base segmentation config (YAML); everything outside the search space comes from it")
    parser.add_argument("--hpo-config", required=True, help="Search space, fidelity and budget (YAML)")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="ifbo", help="HPO algorithm (default: ifbo)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    base = SegmentationConfig.load_from_yaml(args.config)
    hpo = HpoConfig.load_from_yaml(args.hpo_config, base)
    SegmentationStudy(base, hpo, args.optimizer).run()


if __name__ == "__main__":
    main()

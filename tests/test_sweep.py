"""W&B sweep: the sweep definition matches the config, and a trial builds a valid config with its own output dir."""

import itertools
from pathlib import Path

import pytest
import wandb
import yaml

from latent_crew.config import SegmentationConfig, merge, read_yaml
from latent_crew.training import sweep
from latent_crew.utils.runtime import init_wandb

CONFIGS = Path(__file__).parents[1] / "configs"
SWEEP = yaml.safe_load((CONFIGS / "sweeps" / "segmentation.yaml").read_text(encoding="utf-8"))
TRIAL = CONFIGS / "sweeps" / "trial.yaml"


def leaves(parameters: dict, prefix: tuple = ()) -> dict:
    """Flattens the nested sweep `parameters:` block to {(section, key): spec}."""
    out = {}
    for key, spec in parameters.items():
        if "parameters" in spec:
            out.update(leaves(spec["parameters"], prefix + (key,)))
        else:
            out[prefix + (key,)] = spec
    return out


def candidates(spec: dict) -> list:
    """The categorical values, or the two ends of a range."""
    return spec["values"] if "values" in spec else [spec["min"], spec["max"]]


def nested(path: tuple, value) -> dict:
    return {path[0]: nested(path[1:], value)} if len(path) > 1 else {path[0]: value}


def test_sweep_parameters_exist_and_every_extreme_is_valid() -> None:
    base = read_yaml(TRIAL)
    params = leaves(SWEEP["parameters"])
    for section, key in params:
        assert key in base[section], f"{section}.{key} is not a config key"
    # Each value alone, and all lower / all upper ends together, must pass config validation
    for path, spec in params.items():
        for value in candidates(spec):
            SegmentationConfig.from_dict(merge(base, nested(path, value)))
    for pick in (0, -1):
        overrides = {}
        for path, spec in params.items():
            overrides = merge(overrides, nested(path, candidates(spec)[pick]))
        SegmentationConfig.from_dict(merge(base, overrides))


def test_sweep_ranks_by_val_miou_and_trials_are_shorter() -> None:
    assert SWEEP["metric"] == {"name": "metrics/mIoU", "goal": "maximize"}
    trial, baseline = read_yaml(TRIAL), read_yaml(CONFIGS / "segmentation.yaml")
    assert trial["training"]["epochs"] < baseline["training"]["epochs"]
    assert trial["logging"]["log_model"] is False and baseline["logging"]["log_model"] is True
    assert trial["training"]["early_stopping_patience"] == baseline["training"]["early_stopping_patience"]


def test_trial_config_gets_own_output_dir() -> None:
    run_config = merge(read_yaml(TRIAL), {"training": {"lr": 0.02}, "_wandb": {"x": 1}})
    config = sweep.trial_config(run_config, "abc123", "run42")
    assert config.training.lr == 0.02
    assert config.experiment.name == "run42"
    assert Path(config.experiment.output_dir) == Path("checkpoints") / "sweeps" / "abc123"


def test_run_trial_reuses_the_sweep_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    seen = {}

    def fake_train(config: SegmentationConfig) -> Path:
        seen["config"] = config
        seen["same_run"] = init_wandb(config, config.experiment.name) is wandb.run
        wandb.run.finish()
        return tmp_path / "best.pt"

    monkeypatch.setattr(sweep, "train", fake_train)
    sweep.run_trial(TRIAL)
    assert seen["same_run"]
    assert seen["config"].training.epochs == read_yaml(TRIAL)["training"]["epochs"]


def test_swept_values_keep_the_nesting() -> None:
    values = merge(read_yaml(TRIAL), {"training": {"lr": 0.02}, "augmentation": {"scale_range": [0.5, 2.5]}})
    picked = sweep.swept_values(SWEEP["parameters"], values)
    assert set(picked) == {"training", "augmentation"}
    assert picked["training"]["lr"] == 0.02 and picked["augmentation"]["scale_range"] == [0.5, 2.5]
    assert set(itertools.chain(*(picked[s] for s in picked))) == {k for _, k in leaves(SWEEP["parameters"])}


def test_export_best_writes_a_full_length_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    best_config = merge(read_yaml(TRIAL), {"training": {"lr": 0.02, "batch_size": 16}})

    class FakeRun:
        id, config, summary = "run42", best_config, {"metrics/mIoU": 0.5}

    class FakeSweep:
        config = SWEEP

        def best_run(self):
            return FakeRun()

    monkeypatch.setattr(sweep.wandb, "Api", lambda: type("Api", (), {"sweep": lambda self, path: FakeSweep()})())
    dest = sweep.export_best("me/latent-crew/abc", tmp_path / "sweep_best.yaml", base=str(CONFIGS / "segmentation.yaml"))
    config = SegmentationConfig.load_from_yaml(dest)
    assert config.training.lr == 0.02 and config.training.batch_size == 16
    assert config.training.epochs == read_yaml(CONFIGS / "segmentation.yaml")["training"]["epochs"]  # full length
    assert config.logging.log_model and "sweep_best" in config.logging.tags

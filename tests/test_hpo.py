"""HPO smoke tests: config validation and short ifBO / PriorBand studies on synthetic data, CPU, W&B off."""

from pathlib import Path

import pytest
import torch
import yaml

from latent_crew.config import SegmentationConfig
from latent_crew.training.hpo_segmentation import STATE_FILE, HpoConfig, ParamSpec, SegmentationStudy
from test_training import config  # noqa: F401 (fixture)

HPO_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "hpo_segmentation.yaml"
# ifBO needs the FT-PFN weights; the test reuses the copy a real study downloaded instead of fetching ~55 MB
SURROGATE_DIR = Path(__file__).parents[1] / "outputs" / "hpo" / "ifbo_surrogate"


def hpo_raw(tmp_path: Path) -> dict:
    raw = yaml.safe_load(HPO_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(study_name="smoke", output_dir=str(tmp_path / "hpo"), budget_epochs=6)
    raw["fidelity"] = {"min_epochs": 1, "max_epochs": 3}
    raw["ifbo"].update(surrogate_dir=str(SURROGATE_DIR), initial_design_size=2)  # continuations within the small budget
    return raw


def load(raw: dict, base: SegmentationConfig, tmp_path: Path) -> HpoConfig:
    path = tmp_path / "hpo.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    return HpoConfig.load_from_yaml(path, base)


def test_shipped_hpo_config_is_valid() -> None:
    base = SegmentationConfig.load_from_yaml(HPO_CONFIG_PATH.parent / "segmentation.yaml")
    hpo = HpoConfig.load_from_yaml(HPO_CONFIG_PATH, base)
    for spec in hpo.search_space.values():
        assert spec.prior is not None  # PriorBand relies on every prior


@pytest.mark.parametrize("change", [
    lambda raw: raw["search_space"].update({"training.epochs": {"type": "int", "lower": 1, "upper": 5}}),
    lambda raw: raw["search_space"].update({"model.encoder": {"type": "categorical", "choices": ["resnet50"]}}),
    lambda raw: raw["search_space"]["training.lr"].update(prior=1.0),
    lambda raw: raw["search_space"]["training.scheduler"].update(prior="step"),
    lambda raw: raw["search_space"]["training.lr"].update(lower=0.0),
    lambda raw: raw["search_space"].update({"training.freeze_encoder_epochs": {"type": "int", "lower": 0, "upper": 2}}),
    lambda raw: raw.update(budget_epochs=2),
])
def test_invalid_hpo_config_is_rejected(change, config: SegmentationConfig, tmp_path: Path) -> None:
    raw = hpo_raw(tmp_path)
    change(raw)
    with pytest.raises(ValueError):
        load(raw, config, tmp_path)


def test_evaluate_resumes_from_previous_step(config: SegmentationConfig, tmp_path: Path) -> None:
    config.training.freeze_encoder_epochs = 0
    hpo = load(hpo_raw(tmp_path), config, tmp_path)
    study = SegmentationStudy(config, hpo, "priorband")
    params = {name: spec.prior for name, spec in hpo.search_space.items()}
    first, second = tmp_path / "config_1_rung_0", tmp_path / "config_1_rung_1"
    first.mkdir(), second.mkdir()

    result = study.evaluate(first, None, "1_rung_0", epochs=1, **params)
    assert result["cost"] == 1 and 0.0 <= result["objective_to_minimize"] <= 1.0
    iters = torch.load(first / STATE_FILE, weights_only=True)["step"]

    result = study.evaluate(second, first, "1_rung_1", epochs=3, **params)
    state = torch.load(second / STATE_FILE, weights_only=True)
    assert result["cost"] == 2  # only the missing epochs are trained
    assert state["epoch"] == 3 and state["step"] == 3 * iters
    assert state["scheduler"]["last_epoch"] == 3 * iters  # LR schedule continues instead of restarting
    assert not (first / STATE_FILE).exists()
    with pytest.raises(ValueError):
        study.evaluate(tmp_path, second, "1_rung_2", epochs=3, **params)


@pytest.mark.parametrize("optimizer", ["priorband", "ifbo"])
def test_study_runs_and_writes_best_config(optimizer: str, config: SegmentationConfig, tmp_path: Path) -> None:
    if optimizer == "ifbo" and not any(SURROGATE_DIR.rglob("*.pt")):
        pytest.skip(f"FT-PFN surrogate not downloaded to {SURROGATE_DIR}; run an ifBO study once")
    config.training.class_weighting, config.training.freeze_encoder_epochs = "none", 0
    study = SegmentationStudy(config, load(hpo_raw(tmp_path), config, tmp_path), optimizer)
    best_path = study.run()

    trials = sorted(p for p in (study.root / "configs").iterdir() if p.is_dir())
    reports = [yaml.safe_load((t / "report.yaml").read_text(encoding="utf-8")) for t in trials]
    assert all(r["reported_as"] == "success" for r in reports)
    assert sum(r["cost"] for r in reports) >= 6
    newest: dict[str, tuple[int, Path]] = {}
    for trial, report in zip(trials, reports):
        epochs = yaml.safe_load((trial / "config.yaml").read_text(encoding="utf-8"))["epochs"]
        assert report["extra"]["epochs"] == epochs
        config_id = trial.name.split("_")[1]  # config_7_3 (ifBO) or config_7_rung_2 (PriorBand)
        if config_id not in newest or epochs > newest[config_id][0]:
            newest[config_id] = (epochs, trial)
    # Only the newest step of each configuration keeps its training state
    for epochs, trial in newest.values():
        assert torch.load(trial / STATE_FILE, weights_only=True)["epoch"] == epochs
    assert sum((t / STATE_FILE).exists() for t in trials) == len(newest)

    best = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    assert best["training"]["epochs"] == 3
    assert best["model"] == config.to_dict()["model"]
    SegmentationConfig(best)  # trainable with train_segmentation as is


def test_ifbo_rejects_too_many_hyperparameters(config: SegmentationConfig, tmp_path: Path) -> None:
    config.training.freeze_encoder_epochs = 0
    hpo = load(hpo_raw(tmp_path), config, tmp_path)
    hpo.search_space.update({f"training.extra_{i}": ParamSpec("float", 0.0, 1.0) for i in range(4)})  # 7 + 4 > 10
    with pytest.raises(ValueError, match="at most 10"):
        SegmentationStudy(config, hpo, "ifbo")

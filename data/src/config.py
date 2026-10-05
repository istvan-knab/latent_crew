import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: Path
    manifest: Path
    downloads: Path
    raw: Path
    processed: Path
    reports: Path
    datasets: list[str]
    include_raw_nyu: bool
    min_free_space_gb: float
    chunk_size_mb: int
    timeout_s: int
    retries: int


def load_config(data_root: Path | str) -> Config:
    data_root = Path(data_root)
    with open(data_root / "config.toml", "rb") as f:
        toml = tomllib.load(f)

    paths = toml["paths"]
    collect = toml["collect"]
    return Config(
        data_root=data_root,
        manifest=data_root / paths["manifest"],
        downloads=data_root / paths["downloads"],
        raw=data_root / paths["raw"],
        processed=data_root / paths["processed"],
        reports=data_root / paths["reports"],
        datasets=collect["datasets"],
        include_raw_nyu=collect["include_raw_nyu"],
        min_free_space_gb=collect["min_free_space_gb"],
        chunk_size_mb=collect["chunk_size_mb"],
        timeout_s=collect["timeout_s"],
        retries=collect["retries"],
    )
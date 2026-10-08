"""DataCollector: download, verify and prepare the datasets."""
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import requests
import scipy.io
from tqdm import tqdm

from .config import load_config
from .extractors import read_nyu_sample, safe_extract_zip
from .utils import free_space_bytes, get_logger, human_size, sha256_of
from .validators import ValidationError, check_mat, check_size, check_zip

log = get_logger(__name__)


class DataCollector:
    def __init__(self, data_root: Path, manifest_path: Path | None = None) -> None:
        self.data_root = Path(data_root)
        self.cfg = load_config(self.data_root)
        # Fall back to the manifest path from config.toml
        self.manifest_path = Path(manifest_path) if manifest_path else self.cfg.manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        log.info("Manifest loaded: %s", ", ".join(self.manifest))

    def collect(
        self,
        datasets: list[str],
        *,
        verify: bool = True,
        extract: bool = True,
        prepare: bool = True,
        include_raw_nyu: bool = False,
        force: bool = False,
    ) -> dict:
        """Run the full pipeline; returns a summary that is also saved to reports/."""
        if include_raw_nyu:
            log.warning("Raw NYU download is not supported yet, skipping it.")

        free = free_space_bytes(self.cfg.downloads)
        if free < self.cfg.min_free_space_gb * 1024**3:
            raise ValidationError(
                f"Not enough free space: {human_size(free)} "
                f"(need {self.cfg.min_free_space_gb} GiB)"
            )

        summary = {"started": datetime.now().isoformat(timespec="seconds"), "datasets": {}}
        for name in datasets:
            if name not in self.manifest:
                raise ValidationError(f"Unknown dataset: {name}")
            files = {}
            for spec in self.manifest[name]["files"]:
                target = self.cfg.downloads / spec["download"]
                dest = self.cfg.raw / spec["raw"]
                files[spec["id"]] = self._collect_file(spec, target, dest,
                                                       verify, extract, force)
            result = {"files": files}
            if verify and extract:
                result["verify"] = self.verify_dataset(name)
            if prepare:
                result["index"] = str(self.prepare_dataset(name))
            summary["datasets"][name] = result

        summary["finished"] = datetime.now().isoformat(timespec="seconds")
        self.cfg.reports.mkdir(parents=True, exist_ok=True)
        out = self.cfg.reports / "collect_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Summary written to %s", out)
        return summary

    def _collect_file(self, spec: dict, target: Path, dest: Path,
                      verify: bool, extract: bool, force: bool) -> str:
        """Download, verify and extract one manifest entry; returns its status."""
        marker = dest / f".done_{target.name}"
        if force:
            target.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)

        status = "skipped"
        if target.exists():
            try:
                self.verify_download(target, spec)
            except ValidationError as e:
                # Broken or incomplete file: delete it and download again
                log.warning("%s, downloading again", e)
                target.unlink()
                marker.unlink(missing_ok=True)

        if not target.exists():
            self.download_file(spec["url"], target)
            if verify:
                self.verify_download(target, spec)
            status = "downloaded"

        if extract:
            self.extract_archive(target, dest)
        return status

    def download_file(self, url: str, target: Path) -> Path:
        """Download a file; resumes from the .part file after an interruption."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        chunk_size = self.cfg.chunk_size_mb * 1024 * 1024

        for attempt in range(1, self.cfg.retries + 1):
            # Bytes already present from an earlier, interrupted run
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with requests.get(url, headers=headers, stream=True,
                                  timeout=self.cfg.timeout_s) as r:
                    if r.status_code == 416:
                        # 416 = nothing left to send: the .part is complete
                        break
                    r.raise_for_status()
                    if done and r.status_code != 206:
                        # Server ignored the Range header, so start over
                        log.warning("Server does not support resuming, starting over.")
                        done = 0
                    total = done + int(r.headers.get("Content-Length", 0))
                    mode = "ab" if done else "wb"
                    with open(part, mode) as f, tqdm(
                        total=total, initial=done, unit="B", unit_scale=True,
                        unit_divisor=1024, desc=target.name,
                    ) as bar:
                        for data in r.iter_content(chunk_size=chunk_size):
                            f.write(data)
                            bar.update(len(data))
                break
            except requests.RequestException as e:
                log.warning("Download error (attempt %d/%d): %s",
                            attempt, self.cfg.retries, e)
                if attempt == self.cfg.retries:
                    raise
                time.sleep(2 ** attempt)

        # Rename only after the download is complete
        part.replace(target)
        log.info("Downloaded: %s", target)
        return target

    def verify_download(self, target: Path, spec: dict) -> None:
        """Check size, SHA-256 hash and readability of a downloaded file."""
        target = Path(target)
        if not target.exists():
            raise ValidationError(f"Missing file: {target}")

        check_size(target, spec["size_bytes"])

        # Hashing a 7 GB file takes minutes, but it is the only reliable check
        digest = sha256_of(target)
        if spec.get("sha256") is None:
            log.info("%s SHA-256: %s (not in manifest yet)", target.name, digest)
        elif digest != spec["sha256"]:
            raise ValidationError(f"{target.name}: SHA-256 mismatch")

        if spec["kind"] == "zip":
            check_zip(target)
        elif target.suffix == ".mat":
            check_mat(target)

        log.info("OK: %s", target.name)

    def extract_archive(self, archive: Path, destination: Path) -> None:
        """Extract a ZIP (or copy a plain file) into raw/."""
        archive, destination = Path(archive), Path(destination)
        # Marker is written last, so it only exists after a complete run
        marker = destination / f".done_{archive.name}"
        if marker.exists():
            log.info("Already extracted, skipping: %s", archive.name)
            return

        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            count = safe_extract_zip(archive, destination)
            log.info("Extracted: %s (%d members)", archive.name, count)
        else:
            shutil.copy2(archive, destination / archive.name)
            log.info("Copied: %s", archive.name)
        marker.write_text("ok\n", encoding="utf-8")

    def verify_dataset(self, dataset: str) -> dict:
        """Dataset-specific structure checks; raises ValidationError on problems."""
        if dataset == "nyu_depth_v2":
            return self._verify_nyu()
        log.warning("verify_dataset is not implemented yet, skipping: %s", dataset)
        return {}

    def _verify_nyu(self) -> dict:
        raw = self.cfg.raw / "nyu_depth_v2"
        mat_path, splits_path = raw / "nyu_depth_v2_labeled.mat", raw / "splits.mat"
        for path in (mat_path, splits_path):
            if not path.exists():
                raise ValidationError(f"Missing file: {path}")

        with h5py.File(mat_path, "r") as f:
            fields = ["images", "depths", "labels", "instances", "names", "scenes"]
            missing = [k for k in fields if k not in f]
            if missing:
                raise ValidationError(f"{mat_path.name}: missing fields {missing}")

            counts = {k: f[k].shape[0] for k in fields[:4]}
            if len(set(counts.values())) != 1:
                raise ValidationError(f"{mat_path.name}: sample counts differ {counts}")
            n = counts["images"]

            # Fully decode a few samples and check that the shapes match
            for i in (0, n // 2, n - 1):
                s = read_nyu_sample(f, i)
                shapes = {k: v.shape[:2] for k, v in s.items()}
                if len(set(shapes.values())) != 1:
                    raise ValidationError(f"NYU sample {i}: shape mismatch {shapes}")

        splits = scipy.io.loadmat(splits_path)
        train = set(splits["trainNdxs"].ravel())
        test = set(splits["testNdxs"].ravel())
        if train & test:
            raise ValidationError("NYU train/test splits overlap")
        if train | test != set(range(1, n + 1)):
            raise ValidationError("NYU splits do not cover every sample exactly once")

        log.info("NYU OK: %d samples (%d train, %d test)", n, len(train), len(test))
        return {"samples": n, "train": len(train), "test": len(test)}

    def prepare_dataset(self, dataset: str) -> Path:
        raise NotImplementedError
"""DataCollector: download, verify and prepare the datasets."""
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy.io
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .extractors import (SUN_SENSORS, h5_strings, read_nyu_sample, read_sun_instances,
                         read_sun_meta, read_sun_split, safe_extract_zip, sun_depth_to_mm)
from .utils import (copy_file, free_space_bytes, get_logger, human_size, save_image,
                    sha256_of)
from .validators import ValidationError, check_mat, check_size, check_zip

log = get_logger(__name__)

# Index fields that hold file paths (relative to processed/<dataset>/)
PATH_KEYS = ("rgb", "depth", "depth_raw", "semantic", "instance")


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
                # The index is checked after prepare, which may still fix it
                result["verify"] = self.verify_dataset(name, check_index=not prepare)
            if prepare:
                result["index"] = str(self.prepare_dataset(name))
                if verify:
                    result["index_check"] = self._verify_index(name)
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

    def verify_dataset(self, dataset: str, *, check_index: bool = True) -> dict:
        """Dataset-specific structure checks; raises ValidationError on problems."""
        if dataset == "nyu_depth_v2":
            result = self._verify_nyu()
        elif dataset == "sunrgbd":
            result = self._verify_sun()
        else:
            raise ValidationError(f"Unknown dataset: {dataset}")
        if check_index:
            result["index"] = self._verify_index(dataset)
        return result

    def _verify_index(self, dataset: str) -> str:
        """Check that every path listed in index.jsonl exists."""
        out = self.cfg.processed / dataset
        index = out / "index.jsonl"
        if not index.exists():
            log.info("%s: no index.jsonl yet (run --prepare)", dataset)
            return "not prepared"
        missing, count = [], 0
        with open(index, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                count += 1
                missing += [row[k] for k in PATH_KEYS if k in row and not (out / row[k]).exists()]
        if missing:
            raise ValidationError(
                f"{index}: {len(missing)} missing files, e.g. {missing[:5]}")
        log.info("%s index OK: %d samples, all files present", dataset, count)
        return f"ok ({count} samples)"

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

    def _sun_paths(self) -> dict:
        toolbox = self.cfg.raw / "sunrgbd" / "SUNRGBDtoolbox"
        return {
            "meta": toolbox / "Metadata" / "SUNRGBDMeta.mat",
            "seg": toolbox / "Metadata" / "SUNRGBD2Dseg.mat",
            "split": toolbox / "traintestSUNRGBD" / "allsplit.mat",
        }

    def _sun_samples(self) -> list[dict]:
        """One dict per SUN image with absolute file paths and its split."""
        paths = self._sun_paths()
        train, test = read_sun_split(paths["split"])
        samples = []
        for m in read_sun_meta(paths["meta"]):
            folder = self.cfg.raw / "sunrgbd" / m["key"]
            samples.append({
                "key": m["key"],
                "split": "train" if m["key"] in train else "test" if m["key"] in test else None,
                "sensor": SUN_SENSORS.get(m["sensor"], m["sensor"]),
                "folder": folder,
                "rgb": folder / "image" / m["rgb"],
                "depth": folder / "depth_bfx" / m["depth"],  # hole-filled depth
                "depth_raw": folder / "depth" / m["depth"],
                "seg": folder / "seg.mat",
            })
        return samples

    def _verify_sun(self) -> dict:
        paths = self._sun_paths()
        for path in paths.values():
            if not path.exists():
                raise ValidationError(f"Missing file: {path}")

        log.info("Reading SUN metadata (this takes a while)...")
        samples = self._sun_samples()
        n = len(samples)
        train, test = read_sun_split(paths["split"])
        if train & test:
            raise ValidationError(f"SUN train/test splits overlap: {len(train & test)} samples")
        no_split = [s["key"] for s in samples if s["split"] is None]
        if no_split or len(train) + len(test) != n:
            raise ValidationError(
                f"SUN splits do not match metadata ({n} samples, "
                f"{len(train)} + {len(test)} in splits), e.g. {no_split[:3]}")

        # Every sample needs RGB, both depth maps and seg.mat
        missing = [str(s[k]) for s in tqdm(samples, desc="Checking SUN files", unit="sample")
                   for k in ("rgb", "depth", "depth_raw", "seg") if not s[k].exists()]
        if missing:
            raise ValidationError(f"SUN: {len(missing)} missing files, e.g. {missing[:5]}")

        with h5py.File(paths["seg"], "r") as f:
            labels = f["SUNRGBD2Dseg"]["seglabel"][()].ravel()
            if len(labels) != n:
                raise ValidationError(f"SUNRGBD2Dseg has {len(labels)} labels, expected {n}")
            # Fully decode a few samples and check that the shapes match
            for i in (0, n // 2, n - 1):
                s = samples[i]
                shapes = {
                    "rgb": np.array(Image.open(s["rgb"])).shape[:2],
                    "depth": np.array(Image.open(s["depth"])).shape,
                    "semantic": f[labels[i]][()].T.shape,
                    "instance": read_sun_instances(s["seg"]).shape,
                }
                if len(set(shapes.values())) != 1:
                    raise ValidationError(f"SUN sample {s['key']}: shape mismatch {shapes}")

        n_test = sum(s["split"] == "test" for s in samples)
        log.info("SUN OK: %d samples (%d train, %d test)", n, n - n_test, n_test)
        return {"samples": n, "train": n - n_test, "test": n_test}

    def prepare_dataset(self, dataset: str) -> Path:
        """Convert raw/ data into processed/<dataset>/{train,test} plus index.jsonl."""
        if dataset == "nyu_depth_v2":
            return self._prepare_nyu()
        if dataset == "sunrgbd":
            return self._prepare_sun()
        raise ValidationError(f"Unknown dataset: {dataset}")

    def _write_index(self, dataset: str, rows: list[dict], classes: list[str]) -> Path:
        """Write classes.json and index.jsonl (via a temp file) and log the split sizes."""
        out = self.cfg.processed / dataset
        (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
        index = out / "index.jsonl"
        tmp = index.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(index)

        n_test = sum(r["split"] == "test" for r in rows)
        log.info("%s prepared: %d samples (%d train, %d test) -> %s",
                 dataset, len(rows), len(rows) - n_test, n_test, index)
        return index

    def _prepare_nyu(self) -> Path:
        raw = self.cfg.raw / "nyu_depth_v2"
        out = self.cfg.processed / "nyu_depth_v2"
        out.mkdir(parents=True, exist_ok=True)
        # splits.mat uses 1-based MATLAB indices
        test = set(scipy.io.loadmat(raw / "splits.mat")["testNdxs"].ravel().tolist())

        rows = []
        with h5py.File(raw / "nyu_depth_v2_labeled.mat", "r") as f:
            scenes = h5_strings(f, "sceneTypes")
            # Label 0 means unlabeled, label k is names[k - 1]
            classes = ["unlabeled"] + h5_strings(f, "names")

            for i in tqdm(range(len(scenes)), desc="Preparing NYU", unit="sample"):
                num = i + 1
                split = "test" if num in test else "train"
                stem = f"{num:06d}"
                row = {
                    "id": f"nyu_{stem}",
                    "dataset": "nyu_depth_v2",
                    "split": split,
                    "rgb": f"{split}/rgb/{stem}.jpg",
                    "depth": f"{split}/depth/{stem}.png",
                    "semantic": f"{split}/semantic/{stem}.png",
                    "instance": f"{split}/instance/{stem}.png",
                    "scene": scenes[i],
                    "sensor": "kinect_v1",
                }
                rows.append(row)

                # Skip samples that an earlier run already wrote
                if all((out / row[k]).exists() for k in PATH_KEYS if k in row):
                    continue

                s = read_nyu_sample(f, i)
                # Depth is stored in metres; save it as uint16 millimetres
                depth_mm = np.clip(np.round(s["depth"] * 1000), 0, 65535).astype(np.uint16)
                save_image(s["rgb"], out / row["rgb"])
                save_image(depth_mm, out / row["depth"])
                save_image(s["semantic"].astype(np.uint16), out / row["semantic"])
                save_image(s["instance"].astype(np.uint8), out / row["instance"])

        return self._write_index("nyu_depth_v2", rows, classes)

    def _prepare_sun(self) -> Path:
        out = self.cfg.processed / "sunrgbd"
        out.mkdir(parents=True, exist_ok=True)
        log.info("Reading SUN metadata (this takes a while)...")
        samples = self._sun_samples()

        rows = []
        with h5py.File(self._sun_paths()["seg"], "r") as f:
            # 37-class labels: 0 means unlabeled, label k is seg37list[k - 1]
            classes = ["unlabeled"] + h5_strings(f, "seg37list")
            labels = f["SUNRGBD2Dseg"]["seglabel"][()].ravel()

            for i, s in enumerate(tqdm(samples, desc="Preparing SUN", unit="sample")):
                split = s["split"]
                stem = f"{i + 1:06d}"
                scene_file = s["folder"] / "scene.txt"
                row = {
                    "id": f"sun_{stem}",
                    "dataset": "sunrgbd",
                    "split": split,
                    "rgb": f"{split}/rgb/{stem}.jpg",
                    "depth": f"{split}/depth/{stem}.png",
                    "depth_raw": f"{split}/depth_raw/{stem}.png",
                    "semantic": f"{split}/semantic/{stem}.png",
                    "instance": f"{split}/instance/{stem}.png",
                    "scene": scene_file.read_text().strip() if scene_file.exists() else "unknown",
                    "sensor": s["sensor"],
                    "source": s["key"],
                }
                rows.append(row)

                # Skip samples that an earlier run already wrote
                if all((out / row[k]).exists() for k in PATH_KEYS):
                    continue

                # RGB is already JPEG, so copy it instead of re-encoding
                copy_file(s["rgb"], out / row["rgb"])
                for k in ("depth", "depth_raw"):
                    save_image(sun_depth_to_mm(np.array(Image.open(s[k]))), out / row[k])
                # MATLAB v7.3 stores the label maps transposed
                save_image(f[labels[i]][()].T.astype(np.uint8), out / row["semantic"])
                save_image(read_sun_instances(s["seg"]), out / row["instance"])

        return self._write_index("sunrgbd", rows, classes)

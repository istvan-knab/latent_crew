"""DataCollector: az adathalmazok letöltése, ellenőrzése és előkészítése."""
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .config import load_config
from .utils import get_logger, sha256_of
from .validators import ValidationError, check_mat, check_size, check_zip

log = get_logger(__name__)


class DataCollector:
    def __init__(self, data_root: Path, manifest_path: Path | None = None) -> None:
        self.data_root = Path(data_root)
        self.cfg = load_config(self.data_root)
        # Ha nem kapunk külön manifestet, a config.toml-ban megadottat használjuk
        self.manifest_path = Path(manifest_path) if manifest_path else self.cfg.manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        log.info("Manifest betöltve: %s", ", ".join(self.manifest))

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
        raise NotImplementedError

    def download_file(self, url: str, target: Path) -> Path:
        """Letölt egy fájlt; megszakadás után a .part fájlból folytatja."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        chunk_size = self.cfg.chunk_size_mb * 1024 * 1024

        for attempt in range(1, self.cfg.retries + 1):
            # Ennyi bájt van már meg egy korábbi, félbeszakadt letöltésből
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with requests.get(url, headers=headers, stream=True,
                                timeout=self.cfg.timeout_s) as r:
                    if r.status_code == 416:
                        # 416 = "nincs több adat": a .part már teljes
                        break
                    r.raise_for_status()
                    if done and r.status_code != 206:
                        # A szerver nem a kért résztől küldi, ezért elölről kezdjük
                        log.warning("A szerver nem folytatja a letöltést, elölről kezdem.")
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
                log.warning("Letöltési hiba (%d/%d. próba): %s",
                            attempt, self.cfg.retries, e)
                if attempt == self.cfg.retries:
                    raise
                time.sleep(2 ** attempt)

        # Csak a teljes letöltés után kapja meg a végleges nevét
        part.replace(target)
        log.info("Letöltve: %s", target)
        return target

    def verify_download(self, target: Path, spec: dict) -> None:
        """Ellenőrzi a letöltött fájl méretét, hash-ét és olvashatóságát."""
        target = Path(target)
        if not target.exists():
            raise ValidationError(f"Hiányzó fájl: {target}")

        check_size(target, spec["size_bytes"])

        # A hash-számolás egy 7 GB-os fájlnál percekig tart, de csak így biztos
        digest = sha256_of(target)
        if spec.get("sha256") is None:
            log.info("%s SHA-256: %s (még nincs a manifestben)", target.name, digest)
        elif digest != spec["sha256"]:
            raise ValidationError(f"{target.name}: hibás SHA-256 hash")

        if spec["kind"] == "zip":
            check_zip(target)
        elif target.suffix == ".mat":
            check_mat(target)

        log.info("Rendben: %s", target.name)
        
    def extract_archive(self, archive: Path, destination: Path) -> None:
        raise NotImplementedError

    def verify_dataset(self, dataset: str) -> dict:
        raise NotImplementedError

    def prepare_dataset(self, dataset: str) -> Path:
        raise NotImplementedError
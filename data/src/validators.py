"""Integritás- és tartalomellenőrzések."""
import zipfile
from pathlib import Path

import h5py
import scipy.io


class ValidationError(Exception):
    """Hibás vagy hiányos fájl esetén dobjuk."""


def check_size(path: Path, expected: int) -> None:
    """A fájl mérete pontosan egyezik-e a manifestben megadottal."""
    actual = path.stat().st_size
    if actual != expected:
        raise ValidationError(
            f"{path.name}: hibás méret, {actual} bájt (várt: {expected})"
        )


def check_zip(path: Path) -> None:
    """Megnyitható-e a ZIP, és olvasható-e a tartalomjegyzéke."""
    try:
        with zipfile.ZipFile(path) as zf:
            if not zf.namelist():
                raise ValidationError(f"{path.name}: üres ZIP")
    except zipfile.BadZipFile as e:
        raise ValidationError(f"{path.name}: sérült ZIP ({e})") from e


def check_mat(path: Path) -> None:
    """Megnyitható-e a .mat fájl (régi MATLAB vagy HDF5 alapú v7.3)."""
    try:
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
        else:
            keys = [name for name, _, _ in scipy.io.whosmat(path)]
    except Exception as e:
        raise ValidationError(f"{path.name}: nem olvasható .mat ({e})") from e
    if not keys:
        raise ValidationError(f"{path.name}: üres .mat fájl")
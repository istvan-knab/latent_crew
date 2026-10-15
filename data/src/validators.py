"""File integrity and content checks."""
import zipfile
from pathlib import Path

import h5py
import scipy.io


class ValidationError(Exception):
    """Raised for missing, incomplete or corrupt files."""


def check_size(path: Path, expected: int) -> None:
    """Check that the file size matches the manifest exactly."""
    actual = path.stat().st_size
    if actual != expected:
        raise ValidationError(
            f"{path.name}: wrong size, {actual} bytes (expected {expected})"
        )


def check_zip(path: Path) -> None:
    """Check that the ZIP opens and its file list is readable."""
    try:
        with zipfile.ZipFile(path) as zf:
            if not zf.namelist():
                raise ValidationError(f"{path.name}: empty ZIP")
    except zipfile.BadZipFile as e:
        raise ValidationError(f"{path.name}: corrupt ZIP ({e})") from e


def check_mat(path: Path) -> None:
    """Check that the .mat file opens (old MATLAB or HDF5-based v7.3)."""
    try:
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
        else:
            keys = [name for name, _, _ in scipy.io.whosmat(path)]
    except Exception as e:
        raise ValidationError(f"{path.name}: unreadable .mat ({e})") from e
    if not keys:
        raise ValidationError(f"{path.name}: empty .mat file")
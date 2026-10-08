"""ZIP/MAT extraction helpers."""
import zipfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

from .validators import ValidationError

# Sensor folder names in SUN RGB-D -> names used in our index
SUN_SENSORS = {"kv1": "kinect_v1", "kv2": "kinect_v2",
               "realsense": "realsense", "xtion": "xtion"}


def safe_extract_zip(archive: Path, destination: Path) -> int:
    """Extract a ZIP, refusing any member that would land outside destination."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        # Check all paths before writing anything (path traversal guard)
        for m in members:
            target = (destination / m.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValidationError(f"Unsafe path in ZIP: {m.filename}")
        for m in tqdm(members, desc=f"Extracting {Path(archive).name}", unit="file"):
            zf.extract(m, destination)
    return len(members)


def h5_strings(f: h5py.File, key: str) -> list[str]:
    """Decode a MATLAB cell array of strings (v7.3 stores them as references)."""
    return ["".join(map(chr, f[ref][()].ravel())) for ref in f[key][()].ravel()]


def read_nyu_sample(f: h5py.File, i: int) -> dict:
    """Load NYU sample i as arrays in (H, W) / (H, W, 3) layout."""
    # MATLAB v7.3 stores arrays transposed, so swap the axes back
    return {
        "rgb": f["images"][i].transpose(2, 1, 0),
        "depth": f["depths"][i].T,
        "semantic": f["labels"][i].T,
        "instance": f["instances"][i].T,
    }


def sun_key(path: str) -> str:
    """'/n/fs/sun3d/data/SUNRGBD/kv1/x/' -> 'SUNRGBD/kv1/x' (common key for meta and splits)."""
    path = path.rstrip("/")
    return path[path.index("SUNRGBD/"):]


def read_sun_meta(path: Path) -> list[dict]:
    """Per-image info from SUNRGBDMeta.mat, in file order (same order as SUNRGBD2Dseg)."""
    mat = scipy.io.loadmat(path, variable_names=["SUNRGBDMeta"],
                           squeeze_me=True, struct_as_record=False)
    return [
        {"key": sun_key(m.sequenceName), "rgb": m.rgbname,
         "depth": m.depthname, "sensor": m.sensorType}
        for m in np.atleast_1d(mat["SUNRGBDMeta"])
    ]


def read_sun_split(path: Path) -> tuple[set[str], set[str]]:
    """Train and test sample keys from allsplit.mat."""
    mat = scipy.io.loadmat(path, variable_names=["alltrain", "alltest"], squeeze_me=True)
    # atleast_1d: squeeze_me turns a one-element cell into a plain string
    return ({sun_key(p) for p in np.atleast_1d(mat["alltrain"])},
            {sun_key(p) for p in np.atleast_1d(mat["alltest"])})


def sun_depth_to_mm(a: np.ndarray) -> np.ndarray:
    """SUN depth PNGs store millimetres with the 16 bits rotated left by 3."""
    a = a.astype(np.uint16)
    return (a >> 3) | (a << 13)


def read_sun_instances(seg_path: Path) -> np.ndarray:
    """Instance id map from a sample's seg.mat (0 = no object)."""
    seg = scipy.io.loadmat(seg_path, variable_names=["seglabel", "seginstances"])
    label = seg["seglabel"].astype(np.int64)
    if "seginstances" not in seg:
        # Most folders: seglabel already numbers the objects of the image
        return label.astype(np.uint16)
    # NYU folder: seglabel is the class, seginstances counts objects per class
    mask = label > 0
    pair = label * 1000 + seg["seginstances"].astype(np.int64)
    inst = np.zeros(label.shape, np.uint16)
    _, ids = np.unique(pair[mask], return_inverse=True)
    inst[mask] = ids.ravel() + 1
    return inst

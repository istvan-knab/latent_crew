"""Print the structure of a .mat file (both old MATLAB and v7.3/HDF5)."""
import sys

import h5py
import numpy as np
import scipy.io


def show(name, value, indent=0):
    pad = "  " * indent
    if isinstance(value, np.ndarray) and value.dtype.names:  # MATLAB struct
        print(f"{pad}{name}: struct {value.shape}, fields: {value.dtype.names}")
        if value.size:
            for field in value.dtype.names:
                show(field, value.flat[0][field], indent + 1)
    elif isinstance(value, np.ndarray) and value.dtype == object:  # cell array
        print(f"{pad}{name}: cell {value.shape}")
        if value.size:
            show("[0]", value.flat[0], indent + 1)
    elif isinstance(value, np.ndarray):
        sample = value.flat[:3].tolist() if value.size else []
        print(f"{pad}{name}: {value.dtype} {value.shape} e.g. {sample}")
    else:
        print(f"{pad}{name}: {type(value).__name__} {str(value)[:80]}")


def show_h5(f: h5py.File) -> None:
    def visit(name, obj):
        if name.startswith("#refs#"):  # internal storage, thousands of entries
            return
        print(name, getattr(obj, "shape", ""), getattr(obj, "dtype", ""))
        if isinstance(obj, h5py.Dataset) and obj.dtype == h5py.ref_dtype and obj.size:
            # Follow the first reference to see what it points to
            first = f[obj[()].flat[0]]
            if isinstance(first, h5py.Group):
                print(f"  [0] -> struct, fields: {list(first)}")
            else:
                a = first[()]
                print(f"  [0] -> {a.dtype} {a.shape}, unique {np.unique(a)[:15].tolist()}")
    f.visititems(visit)


def main(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            show_h5(f)
    else:
        data = scipy.io.loadmat(path)
        for key, value in data.items():
            if not key.startswith("__"):
                show(key, value)


if __name__ == "__main__":
    main(sys.argv[1])
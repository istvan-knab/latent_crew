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


def main(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            f.visititems(lambda n, o: print(n, getattr(o, "shape", ""), getattr(o, "dtype", "")))
    else:
        data = scipy.io.loadmat(path)
        for key, value in data.items():
            if not key.startswith("__"):
                show(key, value)


if __name__ == "__main__":
    main(sys.argv[1])
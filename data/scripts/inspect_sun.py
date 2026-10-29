"""Print what one SUN RGB-D sample per sensor folder looks like."""
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "raw" / "sunrgbd" / "SUNRGBD"


def describe_png(path: Path) -> None:
    img = Image.open(path)
    a = np.array(img)
    if img.mode.startswith("I"):
        a = a.astype(np.uint16)  # some Pillow versions open 16-bit PNGs as int32
    print(f"  {path.parent.name}/{path.name}: mode {img.mode}, {a.dtype} {a.shape}, "
          f"raw min {a.min()} max {a.max()}")
    if a.dtype == np.uint16:
        # SUN toolbox: rotate the 16 bits right by 3 to get millimetres
        mm = (a >> 3) | (a << 13)
        valid = mm[a > 0]
        print(f"    rotated -> mm: min {valid.min() if valid.size else 0} "
              f"max {mm.max()}, zero pixels {np.mean(a == 0):.1%}")


def describe_sample(sample: Path) -> None:
    print(f"\n=== {sample.relative_to(ROOT)}")
    print("  contents:", sorted(p.name for p in sample.iterdir()))
    for jpg in (sample / "image").glob("*.jpg"):
        img = Image.open(jpg)
        print(f"  image/{jpg.name}: {img.mode} {img.size[0]}x{img.size[1]}")
    for folder in ("depth", "depth_bfx"):
        for png in (sample / folder).glob("*.png"):
            describe_png(png)
    for txt in ("scene.txt", "intrinsics.txt"):
        if (sample / txt).exists():
            print(f"  {txt}: {(sample / txt).read_text().split()}")
    seg = scipy.io.loadmat(sample / "seg.mat")
    for key, value in seg.items():
        if key.startswith("__"):
            continue
        if value.dtype == object:
            names = [str(v.ravel()[0]) if v.size else "" for v in value.ravel()]
            print(f"  seg.mat {key}: cell {value.shape}, e.g. {names[:8]}")
        else:
            print(f"  seg.mat {key}: {value.dtype} {value.shape}, "
                  f"unique values {np.unique(value)[:15].tolist()}")


def main() -> None:
    # First sample (a folder with seg.mat) inside every sensor/source folder
    for sensor in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for source in sorted(p for p in sensor.iterdir() if p.is_dir()):
            seg = next(source.rglob("seg.mat"), None)
            if seg is None:
                print(f"\n=== {source.relative_to(ROOT)}: no seg.mat found")
            else:
                describe_sample(seg.parent)


if __name__ == "__main__":
    main()
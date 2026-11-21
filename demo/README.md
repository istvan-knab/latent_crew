# Demo

Everything needed to show the trained models live. Training and evaluation code stays in `src/latent_crew/`; this folder only holds what the demo runs.

| Path | Content | Status |
|---|---|---|
| `realsense_stream.py`, `realsense_stream.sh` | Live RGB + depth view of a plugged-in Intel RealSense camera (D435i) | done |
| `models/` | The best checkpoints chosen for the demo (`*.pt`, not in Git) | placeholder |
| demo scripts | Live semantic segmentation on the camera stream with a model from `models/` | planned |

## Live camera view

From the repository root (uses the optional `camera` extra, no Docker needed):

```bash
demo/realsense_stream.sh            # q or Esc quits
demo/realsense_stream.sh --sudo     # macOS: if the camera is not found or USB access is refused
```

Supported: macOS 15+ on Apple Silicon, Linux x86_64 and Windows, Python 3.14. Use a USB 3 capable port and cable. Details: the "Demo" section of the [main README](../README.md).

## Models

Copy the chosen checkpoint (e.g. `checkpoints/<experiment>/best.pt`, or download the `segmentation-<run_id>` W&B model artifact) to `models/`. Checkpoints carry their own model spec and class names, so a demo script can rebuild the network from the file alone. Model files are gitignored; note here which run each one comes from.

| File | W&B run / artifact | Val mIoU | Notes |
|---|---|---|---|
| _none yet_ | | | |

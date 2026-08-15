"""E0 - Examine data.  One call -> an organized report on the egg dataset.

report() prints trajectory/frame counts, the per-frame class taxonomy and its imbalance, and the
per-trajectory terminal-outcome breakdown; taxonomy_figure() saves a labelled example grid.
Reads the consolidated feature store (labels) + raw frames for the figure.
"""
from __future__ import annotations
import collections
from pathlib import Path
import numpy as np, h5py
import load_data as D   # NOTE: no global matplotlib.use("Agg") -- would break the notebook's inline backend

NAME = ["safe", "fallen", "flipped", "both"]
OUT = D.FIG


def report():
    frame = np.zeros(4, int); term = collections.Counter(); ntraj = nframe = hasfail = 0
    with h5py.File(D.H5, "r") as hf:
        for hk in sorted(hf.keys()):
            lab = np.asarray(hf[hk]["labels"]); T = len(lab); ntraj += 1; nframe += T
            cls = (lab[:, 0] == 1).astype(int) + 2 * (lab[:, 1] == 1).astype(int)
            for c in range(4): frame[c] += int((cls == c).sum())
            term[NAME[int(lab[-3:, 0].max()) + 2 * int(lab[-3:, 1].max())]] += 1
            hasfail += (cls != 0).any()
    print("=" * 56); print("EGG DATASET REPORT"); print("=" * 56)
    print(f"trajectories: {ntraj}   frames: {nframe}   features: camera_zed_1 DINOv2 ViT-S/14-reg")
    print(f"trajectories with any failure frame: {hasfail} ({hasfail/ntraj:.0%})")
    print("\nper-FRAME class (imbalanced -- episodes end soon after a failure):")
    for c in range(4): print(f"   {NAME[c]:<8} {frame[c]:>7}  ({frame[c]/nframe:.1%})")
    print("\nper-TRAJECTORY terminal outcome:")
    for k in ("safe", "flipped", "both", "fallen"): print(f"   {k:<8} {term[k]}")
    print("\ntaxonomy: (fallen,flipped) -> {(0,0):safe,(1,0):fallen,(0,1):flipped,(1,1):both};"
          " blue/absent yolk = intact/flipped, plate vs table = on-plate/fallen")
    return {"n_traj": ntraj, "n_frame": nframe, "frame_class": dict(zip(NAME, frame.tolist())), "terminal": dict(term)}


def taxonomy_figure(out: Path = OUT / "taxonomy.png"):
    """one example frame per class (camera_rs_0), for the problem-setup figure."""
    import matplotlib.pyplot as plt
    want = {}; info = D.h5_key_to_path()
    with h5py.File(D.H5, "r") as hf:
        for key, (hk, path) in info.items():
            lab = np.asarray(hf[hk]["labels"]); cls = (lab[:, 0] == 1).astype(int) + 2 * (lab[:, 1] == 1).astype(int)
            for c in range(4):
                if c not in want and (cls == c).any(): want[c] = (path, int(np.where(cls == c)[0][len(np.where(cls==c)[0])//2]))
            if len(want) == 4: break
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for c in range(4):
        path, t = want[c]; ax[c].imshow(D.read_frames(path, "camera_rs_0", [t])[0])
        ax[c].set_title(NAME[c], fontsize=12, weight="bold"); ax[c].set_xticks([]); ax[c].set_yticks([])
    fig.suptitle("Egg failure taxonomy", fontsize=13, weight="bold"); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    report(); taxonomy_figure()

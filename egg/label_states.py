"""E2 - v0 state-level labeling  (failure-mode + egg-on-flipper/plate).

Each FRAME gets two labels, giving 5 state classes:
  * failure mode -> GT (fallen / flipped / both / safe) from the dataset;
  * egg location -> VLM per-frame (flipper vs plate)  [the only VLM call in v0].
Combined classes:  flipper > plate > flipped > fallen > both   (Set A ordering).

`label_egg_location()` writes cache/loc_frames.json.
`build_state_dataset()` assembles balanced per-class train/test features (split by trajectory).
`StateMargin` is the single-image margin architecture used by BOTH E3 margins (bt_margin, regression_margin).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, h5py
import torch, torch.nn as nn
from PIL import Image
import load_data as D
from vlm_call import Budget, parse_json

CACHE = D.LABELS          # VLM labels + datasets  (data/egg/labels)
CKPT = D.CKPT             # checkpoints/egg  (re-exported for the E3 margins)
DEV = "cuda:0"
MODEL = "gemini-robotics-er-1.6-preview"

CLASSES = ["flipper", "plate", "flipped", "fallen", "both"]
TARGET_SETS = {   # scalar value per class for each preference set (used by regression_margin / plots)
    "A": {"flipper": 2.0, "plate": 1.0, "flipped": 0.0, "fallen": -1.0, "both": -2.0},   # cautious
    "B": {"plate": 2.0, "flipper": 1.0, "fallen": 0.0, "flipped": -1.0, "both": -2.0},    # speed
}

LOC_PROMPT = ("This image shows a robot serving a fried TOY egg from a metal PAN toward a round PLATE "
 "with a spatula (flipper). Where is the EGG IN THIS IMAGE? "
 "'flipper' = resting on the metal spatula/flipper (carried, above the surface); "
 "'plate' = resting on the round plate; 'pan' = still in the metal pan; 'other' = unclear/mid-transfer. "
 'Reply ONLY compact JSON: {"location":"flipper|plate|pan|other"}')


# ------------------------------------------------------- VLM egg-location labeling
def _candidates():
    """safe frames biased toward mid (flipper) + late-of-safe-ending (plate)."""
    out = []
    with h5py.File(D.H5, "r") as hf:
        info = D.h5_key_to_path()
        hks = list(info.keys()); rng = np.random.default_rng(3); rng.shuffle(hks)
        for key in hks:
            hk, path = info[key]
            lab = np.asarray(hf[hk]["labels"]); T = len(lab)
            if T < 30: continue
            cls = (lab[:, 0] == 1).astype(int) + 2 * (lab[:, 1] == 1).astype(int)
            end_safe = (cls[-3:] == 0).all()
            for f in (0.35, 0.45, 0.55, 0.65):
                t = int(f * (T - 1))
                if cls[t] == 0: out.append((key, path, t, "mid"))
            if end_safe:
                for f in (0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.99):
                    t = int(f * (T - 1))
                    if cls[t] == 0: out.append((key, path, t, "late"))
        rng.shuffle(out)
    return out


def label_egg_location(n_flipper=1000, n_plate=1000, cap_calls=8000, cap_usd=4.0):
    cands = _candidates()
    B = Budget(cap_usd, CACHE / "loc_frames_cost.json"); recs = []; nf = npl = 0
    for key, path, t, ph in cands:
        if (nf >= n_flipper and npl >= n_plate) or B.calls >= cap_calls: break
        if nf >= n_flipper and ph == "mid": continue
        if npl >= n_plate and ph == "late": continue
        im = Image.fromarray(D.read_frames(path, "camera_rs_0", [t])[0])
        r = parse_json(B.call(MODEL, [LOC_PROMPT, im], max_tokens=30, thinking_budget=0)[0]) or {}
        loc = r.get("location")
        if loc == "flipper" and nf >= n_flipper: continue
        if loc == "plate" and npl >= n_plate: continue
        recs.append({"key": key, "t": int(t), "loc": loc})
        nf += loc == "flipper"; npl += loc == "plate"
        if B.calls % 100 == 0: print(f"  calls {B.calls} flipper {nf} plate {npl} ${B.spent:.3f}", flush=True)
    (CACHE / "loc_frames.json").write_text(json.dumps(recs, indent=1))
    import collections
    print(f"SAVED {len(recs)} location labels {dict(collections.Counter(r['loc'] for r in recs))}  ${B.spent:.3f}")


# ------------------------------------------------------- assemble the state dataset
class StateMargin(nn.Module):
    """single-image margin: 384 -> LayerNorm -> MLP -> scalar (no z-scoring)."""
    def __init__(self, d=384, hid=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def _frames(per_traj, class_cap, seed=0):
    loc = json.loads((CACHE / "loc_frames.json").read_text())
    frames = {c: [] for c in CLASSES}
    for r in loc:
        if r["loc"] in ("flipper", "plate"): frames[r["loc"]].append((r["key"], int(r["t"])))
    rng = np.random.default_rng(seed)
    with h5py.File(D.H5, "r") as hf:
        for hk in sorted(hf.keys()):
            lab = np.asarray(hf[hk]["labels"]); cls = (lab[:, 0] == 1).astype(int) + 2 * (lab[:, 1] == 1).astype(int)
            key = hf[hk].attrs["key"]
            for c, cid in (("fallen", 1), ("flipped", 2), ("both", 3)):
                idx = np.where(cls == cid)[0]
                if len(idx):
                    frames[c] += [(key, int(t)) for t in rng.choice(idx, min(per_traj, len(idx)), replace=False)]
    if class_cap > 0:
        for c in CLASSES:
            if len(frames[c]) > class_cap:
                frames[c] = [frames[c][i] for i in rng.choice(len(frames[c]), class_cap, replace=False)]
    return frames


def build_state_dataset(per_traj=40, class_cap=3000, frac_test=0.2, seed=0, device=DEV):
    """-> dict(Xtr, Xte : {class -> (n,384) tensor}, TRf, TEf : {class -> [(key,t)]}). Split by trajectory."""
    frames = _frames(per_traj, class_cap, seed)
    print("frames per class:", {c: len(frames[c]) for c in CLASSES})
    need = {}
    for c in CLASSES:
        for k, t in frames[c]: need.setdefault(k, set()).add(t)
    with h5py.File(D.H5, "r") as hf:
        a2h = {hf[hk].attrs["key"]: hk for hk in hf.keys()}
        cache = {k: {t: np.asarray(hf[a2h[k]]["cam_embd"][t], np.float32).mean(0) for t in ts} for k, ts in need.items()}
    keys = sorted({k for c in CLASSES for k, _ in frames[c]})
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    trk = set(keys[int(frac_test * len(keys)):])
    TRf = {c: [(k, t) for k, t in frames[c] if k in trk] for c in CLASSES}
    TEf = {c: [(k, t) for k, t in frames[c] if k not in trk] for c in CLASSES}
    def feat(fr): return np.stack([cache[k][t] for k, t in fr]) if fr else np.zeros((0, 384), np.float32)
    Xtr = {c: torch.tensor(feat(TRf[c]), dtype=torch.float32, device=device) for c in CLASSES}
    Xte = {c: torch.tensor(feat(TEf[c]), dtype=torch.float32, device=device) for c in CLASSES}
    (CACHE / "state_frames.json").write_text(json.dumps({"train": TRf, "test": TEf}))
    return {"Xtr": Xtr, "Xte": Xte, "TRf": TRf, "TEf": TEf}


if __name__ == "__main__":
    label_egg_location()

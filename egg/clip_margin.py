"""E5 - Bradley-Terry failure margin on 5-frame clips (nuanced preference).

Clip = 5 frames of DINO-WM latents (cam_embd, mean-pooled over patches -> 384/frame).
`concat` head: shared per-frame phi(384->64), concat the 5 -> LayerNorm -> MLP -> scalar
(captures the temporal flip transition that mean-over-frames washes out). NO z-scoring.
Trained with BT loss on the clip pairs from label_pairs (cache/pref_clip_dataset.json).
Set B (speed) is derived by reversing the R2/R3 axis directions of the Set-A labels.
Saves checkpoints/clip_margin_{SET}_{mode}.pt.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, h5py
import load_data as D

CACHE = D.LABELS; CKPT = D.CKPT; DEV = "cuda:0"; K = 5
DATASET = CACHE / "pref_clip_dataset.json"
REVERSED = {"motion", "position", "release"}     # + failure_mode when it's flip-vs-fall (axis != 'both')


class ClipMargin(nn.Module):
    """mode='concat': per-frame phi then concat the K; mode='mean': mean-pool the K frames."""
    def __init__(self, mode="concat", d=384, proj=64, hid=128, k=K, p=0.3, ln=True):
        super().__init__()
        self.mode = mode
        din = proj * k if mode == "concat" else d
        if mode == "concat": self.phi = nn.Linear(d, proj)
        layers = ([nn.LayerNorm(din)] if ln else []) + [nn.Linear(din, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                       # x: (B,k,d)
        h = torch.relu(self.phi(x)).flatten(1) if self.mode == "concat" else x.mean(1)
        return self.net(h).squeeze(-1)


def load_feats(data):
    """(key, idx-of-5) -> (5,384) mean-pooled DINO features, cached per key."""
    need = {}
    for d in data:
        for c in (d["clipA"], d["clipB"]): need.setdefault(c[0], set()).update(c[1])
    with h5py.File(D.H5, "r") as hf:
        a2h = {hf[hk].attrs["key"]: hk for hk in hf.keys()}
        feat = {k: {t: np.asarray(hf[a2h[k]]["cam_embd"][t], np.float32).mean(0) for t in ts} for k, ts in need.items()}
    return lambda c: np.stack([feat[c[0]][t] for t in c[1]])


def derive_set_b(data):
    """reverse the preferred side on axes Set B opposes (motion/position/release, flip-vs-fall)."""
    out = []
    for d in data:
        rule, axis = d.get("deciding_rule"), d.get("axis")
        flip = rule in REVERSED or (rule == "failure_mode" and axis != "both")
        e = dict(d); e["better"] = ("B" if d["better"] == "A" else "A") if flip else d["better"]; out.append(e)
    return out


def train(set_name="A", mode="concat", epochs=200, data=None):
    data = data if data is not None else json.loads(DATASET.read_text())
    if set_name == "B": data = derive_set_b(data)
    clip = load_feats(data)
    pref = np.stack([clip(d["clipA"] if d["better"] == "A" else d["clipB"]) for d in data])
    disp = np.stack([clip(d["clipB"] if d["better"] == "A" else d["clipA"]) for d in data])
    Pf = torch.tensor(pref, dtype=torch.float32, device=DEV); Df = torch.tensor(disp, dtype=torch.float32, device=DEV)
    N = len(data); rng = np.random.default_rng(0); perm = rng.permutation(N); nte = N // 5
    te, tr = torch.tensor(perm[:nte], device=DEV), torch.tensor(perm[nte:], device=DEV)
    torch.manual_seed(0); m = ClipMargin(mode).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(epochs):
        m.train(); p = tr[torch.randperm(len(tr), device=DEV)]
        for s in range(0, len(p), 128):
            b = p[s:s + 128]
            loss = torch.nn.functional.softplus(m(Df[b]) - m(Pf[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): acc = (m(Pf[te]) > m(Df[te])).float().mean().item()
    torch.save({"state_dict": m.state_dict(), "mode": mode, "set": set_name}, CKPT / f"clip_margin_{set_name}_{mode}.pt")
    print(f"[clip {set_name}/{mode}] {N} pairs  held-out pair-acc={acc:.3f}")
    return m


def load(set_name="A", mode="concat"):
    b = torch.load(CKPT / f"clip_margin_{set_name}_{mode}.pt", map_location=DEV, weights_only=False)
    m = ClipMargin(b["mode"]).to(DEV); m.load_state_dict(b["state_dict"]); m.eval()
    return m


if __name__ == "__main__":
    for s, mode in (("A", "concat"), ("A", "mean"), ("B", "concat")):
        train(s, mode)

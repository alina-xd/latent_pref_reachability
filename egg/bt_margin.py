"""E3a - Bradley-Terry state-level failure margin.

Preference pairs (higher-target class ≻ lower-target class), 100k pairs for EACH of the 10 ordered
class-pairs, softplus BT loss. Output normalized to [-2,2] with simple min-max over train frames.
Same architecture as the regression margin (StateMargin); the two differ only in the loss.
Saves checkpoints/margin_bt_{SET}.pt (state_dict + min/max). Run to train both preference sets.
"""
import numpy as np, torch
from label_states import build_state_dataset, StateMargin, CLASSES, TARGET_SETS, CKPT, DEV



def train(set_name="A", per_pair=100000, epochs=60, data=None):
    T = TARGET_SETS[set_name]
    d = data or build_state_dataset()
    Xtr, Xte = d["Xtr"], d["Xte"]
    Xall = torch.cat([Xtr[c] for c in CLASSES]); off = {}; o = 0
    for c in CLASSES: off[c] = o; o += Xtr[c].shape[0]
    gp = [(hi, lo) for hi in CLASSES for lo in CLASSES if T[hi] > T[lo]]      # 10 ordered class-pairs
    rng = np.random.default_rng(0)
    hi_ix = torch.tensor(np.concatenate([off[hi] + rng.integers(Xtr[hi].shape[0], size=per_pair) for hi, lo in gp]), device=DEV)
    lo_ix = torch.tensor(np.concatenate([off[lo] + rng.integers(Xtr[lo].shape[0], size=per_pair) for hi, lo in gp]), device=DEV)
    print(f"[BT set {set_name}] {per_pair}/class-pair x {len(gp)} = {len(hi_ix)} pairs")

    torch.manual_seed(0); m = StateMargin().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3); N = len(hi_ix); BS = 1024
    for ep in range(epochs):
        m.train(); perm = torch.randperm(N, device=DEV)
        for s in range(0, N, BS):
            b = perm[s:s + BS]
            loss = torch.nn.functional.softplus(m(Xall[lo_ix[b]]) - m(Xall[hi_ix[b]])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): allv = torch.cat([m(Xtr[c]) for c in CLASSES]).cpu().numpy()
    lo, hi = float(allv.min()), float(allv.max())
    torch.save({"state_dict": m.state_dict(), "lo": lo, "hi": hi, "set": set_name}, CKPT / f"margin_bt_{set_name}.pt")
    _report(set_name, m, lambda v: 4 * (v - lo) / (hi - lo) - 2, Xte, T)
    return m


def load(set_name="A"):
    b = torch.load(CKPT / f"margin_bt_{set_name}.pt", map_location=DEV, weights_only=False)
    m = StateMargin().to(DEV); m.load_state_dict(b["state_dict"]); m.eval()
    lo, hi = b["lo"], b["hi"]
    return m, (lambda v: 4 * (v - lo) / (hi - lo) - 2)


def _report(set_name, m, norm, Xte, T):
    def sc(c):
        with torch.no_grad(): return norm(m(Xte[c]).cpu().numpy())
    auc = lambda a, b: (a[:, None] > b[None, :]).mean()
    fm = ("flipped", "fallen") if T["flipped"] > T["fallen"] else ("fallen", "flipped")
    lc = ("flipper", "plate") if T["flipper"] > T["plate"] else ("plate", "flipper")
    print(f"[BT set {set_name}] held-out  {fm[0]}>{fm[1]}={auc(sc(fm[0]),sc(fm[1])):.3f}   {lc[0]}>{lc[1]}={auc(sc(lc[0]),sc(lc[1])):.3f}")


if __name__ == "__main__":
    d = build_state_dataset()
    for s in ("A", "B"): train(s, data=d)

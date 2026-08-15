"""E3b - Regression state-level failure margin.

Same architecture (StateMargin) as the BT margin, but trained with (class-weighted) MSE to the
per-class integer targets of the preference set, e.g. Set A: flipper +2, plate +1, flipped 0,
fallen -1, both -2. Output lands near [-2,2] by construction (no normalization needed).
Saves checkpoints/margin_reg_{SET}.pt. Run to train both preference sets.
"""
import numpy as np, torch
from label_states import build_state_dataset, StateMargin, CLASSES, TARGET_SETS, CKPT, DEV



def train(set_name="A", reg_n=800, epochs=300, data=None):
    """reg_n>0 -> balance to reg_n/class (plate-limited); 0 -> all frames with class-weighted MSE."""
    T = TARGET_SETS[set_name]
    d = data or build_state_dataset()
    Xtr, Xte = d["Xtr"], d["Xte"]; rng = np.random.default_rng(0)
    if reg_n > 0:
        nb = min([reg_n] + [Xtr[c].shape[0] for c in CLASSES])
        idx = {c: torch.tensor(rng.choice(Xtr[c].shape[0], nb, replace=False), device=DEV) for c in CLASSES}
        Xr = torch.cat([Xtr[c][idx[c]] for c in CLASSES]); cnt = np.array([nb] * len(CLASSES), float)
        yr = torch.cat([torch.full((nb,), T[c], device=DEV) for c in CLASSES])
        print(f"[reg set {set_name}] balanced {nb}/class = {nb*len(CLASSES)} frames")
    else:
        Xr = torch.cat([Xtr[c] for c in CLASSES]); cnt = np.array([Xtr[c].shape[0] for c in CLASSES], float)
        yr = torch.cat([torch.full((Xtr[c].shape[0],), T[c], device=DEV) for c in CLASSES])
        print(f"[reg set {set_name}] all {int(cnt.sum())} frames, class-weighted")
    cw = cnt.sum() / (len(CLASSES) * cnt)                                   # =1 each when balanced
    wr = torch.cat([torch.full((int(cnt[i]),), float(cw[i]), device=DEV) for i in range(len(CLASSES))])

    torch.manual_seed(0); m = StateMargin().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3); Nr = len(yr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(Nr, device=DEV)
        for s in range(0, Nr, 256):
            b = perm[s:s + 256]
            loss = (wr[b] * (m(Xr[b]) - yr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    torch.save({"state_dict": m.state_dict(), "set": set_name}, CKPT / f"margin_reg_{set_name}.pt")
    _report(set_name, m, Xte, T)
    return m


def load(set_name="A"):
    b = torch.load(CKPT / f"margin_reg_{set_name}.pt", map_location=DEV, weights_only=False)
    m = StateMargin().to(DEV); m.load_state_dict(b["state_dict"]); m.eval()
    return m, None                                                          # no normalization


def _report(set_name, m, Xte, T):
    def sc(c):
        with torch.no_grad(): return m(Xte[c]).cpu().numpy()
    auc = lambda a, b: (a[:, None] > b[None, :]).mean()
    fm = ("flipped", "fallen") if T["flipped"] > T["fallen"] else ("fallen", "flipped")
    lc = ("flipper", "plate") if T["flipper"] > T["plate"] else ("plate", "flipper")
    print(f"[reg set {set_name}] held-out  {fm[0]}>{fm[1]}={auc(sc(fm[0]),sc(fm[1])):.3f}   {lc[0]}>{lc[1]}={auc(sc(lc[0]),sc(lc[1])):.3f}")


if __name__ == "__main__":
    d = build_state_dataset()
    for s in ("A", "B"): train(s, data=d)

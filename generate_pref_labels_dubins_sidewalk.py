"""Generate comparative PREFERENCE labels for learning the failure margin.

Instead of regressing the scalar weighted signed-distance margin, we learn it
from PAIRWISE preferences: for a pair of states (A, B), the one with the higher
ground-truth weighted margin (i.e. "safer") is preferred. Pairs cover all three
regimes:
  - safe-safe        (ss): both margins > 0
  - failure-failure  (ff): both margins < 0
  - failure-safe     (fs): one > 0, one < 0

The rendered observations match the world-model training data (same render_state,
dpi=128), so each state can be encoded through the WM and scored by a learned
margin head f(latent).

Output pickle (memory-efficient: a pooled set of states + index pairs):
  imgs        (M,H,W,3) uint8   rendered observation per state
  obs_state   (M,2)             [cos theta, sin theta]
  priv        (M,3)             [x, y, theta]
  margin      (M,)              ground-truth weighted margin (>0 safe / <0 fail)
  pairs       (N,2) int         (i, j) indices into the pool
  label       (N,) int          1 if state i preferred (margin_i > margin_j) else 0
  category    (N,) '<U2'        'ss' / 'ff' / 'fs'
  pref_profile, weights, R, SW

Downstream Bradley-Terry training over the pairs:
  logit = f(latent_i) - f(latent_j)
  loss  = BCEWithLogits(logit, label)          # label = 1 iff i is safer

Usage (from repo root):
  python generate_pref_labels_dubins_sidewalk.py --pref_profile set_a
  python generate_pref_labels_dubins_sidewalk.py --pref_profile set_b \
      --n_states 5000 --n_pairs 30000 --sample_fig pref_samples_setb.png
"""
import argparse
import os
import pickle
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_data_dubins_sidewalk as gen

R, SW = gen.OBS_R, gen.SIDEWALK_Y

# Ground-truth weighted signed-distance margin -- mirrors the canonical definition
# in eais_hw2/dreamerv3-torch/dreamer_offline_sidewalk.py (keep in sync).
PREF_PROFILES = {
    "set_a": {"center": 3.0, "upper_sw": 1.0, "lower_sw": 1.0, "upper_safe": 3.0, "lower_safe": 0.5},
    "set_b": {"center": 1.0, "upper_sw": 3.0, "lower_sw": 3.0, "upper_safe": 1.0, "lower_safe": 3.0},
}


def weighted_margin_np(x, y, w, R, SW):
    l_center = float(np.hypot(x, y)) - R      # >0 outside obstacle
    l_side = SW - abs(y)                       # >0 inside the drivable band
    m = min(l_center, l_side)
    if m < 0:                                  # unsafe
        if l_center < 0:                       # inside the obstacle
            m = l_center * w["center"]
        elif y > SW:                           # upper sidewalk
            m = l_side * w["upper_sw"]
        elif y < -SW:                          # lower sidewalk
            m = l_side * w["lower_sw"]
    else:                                      # safe region
        m = m * (w["upper_safe"] if y > 0 else w["lower_safe"])
    return m


def sample_states(n, weights, rng, speed, dt, dpi, emoji, dom=1.5):
    """Render n states, sampled to cover the safe region and BOTH failure types
    (obstacle interior + off-sidewalk), so ff/fs pairs are well represented."""
    imgs, priv, margin = [], [], []
    for _ in range(n):
        u = rng.random(); th = rng.uniform(0, 2 * np.pi)
        if u < 0.5:                                          # uniform over the domain
            x, y = rng.uniform(-dom, dom, 2)
        elif u < 0.75:                                       # inside the obstacle (failure)
            r = R * np.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
            x, y = r * np.cos(a), r * np.sin(a)
        else:                                                # off a sidewalk edge (failure)
            x = rng.uniform(-dom, dom); y = rng.choice([-1.0, 1.0]) * rng.uniform(SW, dom)
        imgs.append(gen.render_state(np.array([x, y, th], np.float32), speed, dt, emoji, dpi).astype(np.uint8))
        priv.append([x, y, th]); margin.append(weighted_margin_np(x, y, weights, R, SW))
    priv = np.array(priv, np.float32); margin = np.array(margin, np.float32)
    obs_state = np.stack([np.cos(priv[:, 2]), np.sin(priv[:, 2])], 1).astype(np.float32)
    return np.array(imgs), obs_state, priv, margin


def make_pairs(margin, n_pairs, rng, min_gap=0.0):
    """Form ~n_pairs/3 each of safe-safe, failure-failure, failure-safe pairs.
    label = 1 iff the FIRST index has the higher (safer) ground-truth margin."""
    safe = np.where(margin > 0)[0]; fail = np.where(margin < 0)[0]
    assert len(safe) > 1 and len(fail) > 1, f"need >1 safe and >1 fail states (safe={len(safe)}, fail={len(fail)})"
    per = n_pairs // 3
    pairs, labels, cats = [], [], []

    def draw(cat, pool_i, pool_j):
        c = 0
        while c < per:
            i = int(rng.choice(pool_i)); j = int(rng.choice(pool_j))
            if i == j or abs(margin[i] - margin[j]) <= min_gap:
                continue
            if rng.random() < 0.5:                 # randomize which of the pair is "A"
                i, j = j, i
            pairs.append((i, j)); labels.append(int(margin[i] > margin[j])); cats.append(cat); c += 1

    draw("ss", safe, safe)
    draw("ff", fail, fail)
    draw("fs", fail, safe)
    return np.array(pairs, np.int64), np.array(labels, np.int64), np.array(cats)


def save_sample_fig(out, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cat, pairs, margin, label, imgs = out["category"], out["pairs"], out["margin"], out["label"], out["imgs"]
    names = {"ss": "safe-safe", "ff": "failure-failure", "fs": "failure-safe"}
    fig, axes = plt.subplots(3, 2, figsize=(5, 7.6))
    for r, c in enumerate(["ss", "ff", "fs"]):
        idxs = np.where(cat == c)[0]
        pr = pairs[idxs[0]]
        for col, k in enumerate(pr):
            axes[r, col].imshow(imgs[k]); axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            preferred = (label[idxs[0]] == 1 and col == 0) or (label[idxs[0]] == 0 and col == 1)
            axes[r, col].set_title(f"{names[c]}  {'A' if col == 0 else 'B'}   m={margin[k]:+.2f}"
                                   + ("   ← PREFERRED" if preferred else ""),
                                   fontsize=8, color=("green" if preferred else "black"))
    fig.suptitle(f"preference pairs ({out['pref_profile']})  green = safer (preferred)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"[pref-gen] sample figure -> {path}")


def main(a):
    rng = np.random.default_rng(a.seed)
    emoji = gen.load_tree_emoji()
    weights = PREF_PROFILES[a.pref_profile]
    print(f"[pref-gen] profile={a.pref_profile}  weights={weights}")
    imgs, obs_state, priv, margin = sample_states(a.n_states, weights, rng, a.speed, a.dt, a.dpi, emoji)
    print(f"[pref-gen] pool: {len(imgs)} states  safe={int((margin>0).sum())} fail={int((margin<0).sum())}  "
          f"margin range [{margin.min():+.2f}, {margin.max():+.2f}]")
    pairs, label, cat = make_pairs(margin, a.n_pairs, rng, a.min_gap)
    print(f"[pref-gen] pairs: {len(pairs)}  ss={int((cat=='ss').sum())} ff={int((cat=='ff').sum())} "
          f"fs={int((cat=='fs').sum())}  P(A preferred)={label.mean():.3f}")
    out = {"imgs": imgs, "obs_state": obs_state, "priv": priv, "margin": margin,
           "pairs": pairs, "label": label, "category": cat,
           "pref_profile": a.pref_profile, "weights": weights, "R": R, "SW": SW}
    op = a.out or f"data/pref_labels_{a.pref_profile}.pkl"
    with open(op, "wb") as f:
        pickle.dump(out, f)
    print(f"[pref-gen] saved -> {op}  ({os.path.getsize(op)/1e6:.1f} MB)")
    if a.sample_fig:
        save_sample_fig(out, a.sample_fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref_profile", default="set_a", choices=list(PREF_PROFILES))
    ap.add_argument("--n_states", type=int, default=4000, help="pool size (# rendered states)")
    ap.add_argument("--n_pairs", type=int, default=24000, help="# preference pairs (~1/3 per category)")
    ap.add_argument("--dpi", type=int, default=128, help="image size (match WM training data)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--min_gap", type=float, default=0.0, help="skip pairs with |margin_i-margin_j| <= this (drop near-ties)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output .pkl path")
    ap.add_argument("--sample_fig", default=None, help="save a 3-category sample figure here")
    main(ap.parse_args())

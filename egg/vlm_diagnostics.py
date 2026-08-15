"""E1 - VLM label diagnostics.

Two audits of the VLM as a labeler, scored against the ground-truth failure labels:
  * classify() -- per frame, recover (fallen, flipped);
  * preference() -- per pair, judge which state is safer (safe > single-failure > both).
Small stratified samples (Gemini ER 1.6). Establishes that the VLM is a usable labeler before we
build the margin datasets on its outputs.
"""
from __future__ import annotations
import numpy as np, h5py
from PIL import Image
import load_data as D
from vlm_call import Budget, parse_json

MODEL = "gemini-robotics-er-1.6-preview"
NAME = ["safe", "fallen", "flipped", "both"]
CLS_PROMPT = ("A robot serves a fried TOY egg (pan -> plate) with a spatula. In THIS image: is the egg "
 "FALLEN on the white table (not pan/plate)? is it FLIPPED (no round yolk, a pale blob)? "
 'Reply ONLY JSON: {"fallen":0|1,"flipped":0|1}')
PREF_PROMPT = ("Two images of a robot serving a fried TOY egg. Which is the SAFER state? "
 "safe (egg intact on pan/flipper/plate) > single failure (fallen OR flipped) > both. "
 'Reply ONLY JSON: {"safer":"A|B"}')


def _sample(n_per_class, seed=0):
    """-> {class -> [(path, t)]} from GT labels."""
    rng = np.random.default_rng(seed); out = {c: [] for c in range(4)}
    info = D.h5_key_to_path()
    with h5py.File(D.H5, "r") as hf:
        for key, (hk, path) in info.items():
            lab = np.asarray(hf[hk]["labels"]); cls = (lab[:, 0] == 1).astype(int) + 2 * (lab[:, 1] == 1).astype(int)
            for c in range(4):
                idx = np.where(cls == c)[0]
                if len(idx) and len(out[c]) < n_per_class * 3:
                    out[c].append((path, int(rng.choice(idx))))
    return {c: [out[c][i] for i in rng.choice(len(out[c]), min(n_per_class, len(out[c])), replace=False)] for c in range(4)}


def classify(n_per_class=15, cap_usd=1.0):
    samp = _sample(n_per_class); B = Budget(cap_usd); correct = tot = 0
    for c, frs in samp.items():
        for path, t in frs:
            im = Image.fromarray(D.read_frames(path, "camera_rs_0", [t])[0])
            r = parse_json(B.call(MODEL, [CLS_PROMPT, im], max_tokens=30, thinking_budget=0)[0]) or {}
            pred = int(r.get("fallen", 0) == 1) + 2 * int(r.get("flipped", 0) == 1)
            correct += pred == c; tot += 1
    print(f"[classify] {correct}/{tot} = {correct/max(tot,1):.2f} frame-level accuracy  (${B.spent:.3f})")


def preference(n_pairs=30, cap_usd=1.0):
    samp = _sample(20); rank = {0: 3, 1: 1, 2: 1, 3: 0}      # safe > single > both
    flat = [(c, p, t) for c, frs in samp.items() for p, t in frs]
    rng = np.random.default_rng(1); B = Budget(cap_usd); correct = tot = 0
    for _ in range(n_pairs):
        (ca, pa, ta), (cb, pb, tb) = flat[rng.integers(len(flat))], flat[rng.integers(len(flat))]
        if rank[ca] == rank[cb]: continue
        ims = [Image.fromarray(D.read_frames(p, "camera_rs_0", [t])[0]) for p, t in ((pa, ta), (pb, tb))]
        r = parse_json(B.call(MODEL, [PREF_PROMPT, "A:", ims[0], "B:", ims[1]], max_tokens=20, thinking_budget=0)[0]) or {}
        gt = "A" if rank[ca] > rank[cb] else "B"
        correct += r.get("safer") == gt; tot += 1
    print(f"[preference] {correct}/{tot} = {correct/max(tot,1):.2f} pair accuracy  (${B.spent:.3f})")


if __name__ == "__main__":
    classify(); preference()

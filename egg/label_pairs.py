"""E4 - Trajectory-level (clip) labeling with the nuanced lexicographic preference.

5-frame clips (stride 5). Preference (Set A, cautious), highest priority first:
  R1 SAFETY      safe ≻ any failure
  R2 FAILURE MODE  both worst; else flipped-on-plate ≻ fallen-on-table
  R3 STYLE       position (center≻edge, only if both plated) > motion (small≻large travel)
                 > release (gentle-low ≻ drop-high)

Pipeline (all append to cache/pref_clip_dataset.json):
  1. collect_vlm_pairs()  -> 500 non-equal VLM-labeled pairs (Gemini ER 1.6)
  2. add_both_pairs()     -> explicit `both`-is-worst pairs        (GT, no VLM)
  3. add_crosslevel_pairs() -> dispreferred-safe ≻ failure R1 pins (GT, no VLM)
Set B (speed) is derived by reversing the R2/R3 axes (see clip_margin.derive_set_b).
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np, h5py
from PIL import Image
import load_data as D
from vlm_call import Budget, parse_json

CACHE = D.LABELS
MODEL = "gemini-robotics-er-1.6-preview"
DATASET = CACHE / "pref_clip_dataset.json"
S = 5                                                          # frame stride within a clip
PLO = np.array([0.509, -0.008]); PHI = np.array([0.631, 0.166]); PC = (PLO + PHI) / 2   # plate bbox / center
_k2p = {f"{p.parent.name}/{p.stem}": p for p in D.list_trajectories()}

PROMPT = ('Compare two short video clips of a robot serving a fried TOY egg with a spatula (flipper), '
 'moving it from a metal PAN toward a round PLATE. Clip A = the first 5 images, Clip B = the next 5 '
 'images (both in time order). Color legend: a round yolk visible = sunny-side-up (intact/good); a '
 'plain pale blob = flipped upside-down (ruined). "fallen"=egg dropped onto the TABLE; "flipped"=egg '
 'turned over; "both"=fallen AND flipped; "safe"=egg stays intact on the pan/flipper/plate.\n\n'
 'FIRST judge EACH clip\'s OUTCOME. THEN apply the ranking. Evaluate EVERY rule -- do NOT answer '
 '"safety" unless one clip actually FAILS while the other stays SAFE.\n'
 'R1 SAFETY: safe beats fail. If BOTH safe, R1 is a TIE -> continue to R3. If BOTH fail, skip to R2.\n'
 'R2 FAILURE MODE (both fail): "both" is worst; else prefer FLIPPED-ON-THE-PLATE over FALLEN-ON-THE-TABLE.\n'
 'R3 (both safe), in order, moving on only on a genuine tie:\n'
 '  R3a POSITION -- ONLY if BOTH clips already have the egg resting ON THE PLATE; else "na". Prefer CENTER over EDGE.\n'
 '  R3b MOTION -- prefer SMALL spatula travel (slow) over LARGE travel (fast).\n'
 '  R3c RELEASE -- prefer spatula LOW touching the plate (gentle) over HIGH with the egg falling (drop).\n'
 'Answer "equal" ONLY if tied on R1 AND R3a AND R3b AND R3c.\n'
 'Reply ONLY compact JSON no markdown: {"A_outcome":"safe|fallen|flipped|both","B_outcome":"safe|fallen|'
 'flipped|both","r3a_position":"A|B|tie|na","r3b_motion":"A|B|tie|na","r3c_release":"A|B|tie|na",'
 '"better":"A|B|equal","deciding_rule":"safety|failure_mode|position|motion|release"}')


# ------------------------------------------------------- clip pools (from GT labels + ee signals)
def build_pools():
    """-> flip, fall (onset-centered failure clips); safe (windows tagged with disp/tz/on_plate/dc)."""
    flip, fall, safe = [], [], []
    with h5py.File(D.H5, "r") as hf:
        for hk in sorted(hf.keys()):
            key = hf[hk].attrs["key"]
            if key not in _k2p: continue
            lab = np.asarray(hf[hk]["labels"]); ee = D.read_lowdim(_k2p[key])["ee_pos"]; T = len(lab)
            fa = (lab[:, 0] == 1) & (lab[:, 1] == 0); fl = (lab[:, 1] == 1) & (lab[:, 0] == 0)
            for mask, pool in ((fl, flip), (fa, fall)):
                if mask.any():
                    on = int(np.argmax(mask)); idx = [on - 2 * S, on - S, on, on + S, on + 2 * S]
                    if idx[0] >= 0 and idx[-1] < T and mask[on:on + 2].all(): pool.append([key, idx])
            rng = np.random.default_rng(abs(hash(key)) % 2 ** 32)
            for _ in range(6):
                s = int(rng.integers(1, max(2, T - 4 * S - 1))); idx = [s + S * i for i in range(5)]
                if idx[-1] >= T or not ((lab[idx, 0] == 0).all() and (lab[idx, 1] == 0).all()): continue
                txy = ee[idx[-1], :2]
                safe.append({"key": key, "idx": idx, "disp": float(np.linalg.norm(ee[idx[-1], :2] - ee[idx[0], :2])),
                             "tz": float(ee[idx[-1], 2]), "on_plate": bool(np.all(txy >= PLO - .03) and np.all(txy <= PHI + .03)),
                             "dc": float(np.linalg.norm(txy - PC))})
    return flip, fall, safe


def both_clips():
    out = []
    with h5py.File(D.H5, "r") as hf:
        for hk in sorted(hf.keys()):
            key = hf[hk].attrs["key"]
            if key not in _k2p: continue
            lab = np.asarray(hf[hk]["labels"]); T = len(lab); bo = (lab[:, 0] == 1) & (lab[:, 1] == 1)
            if bo.any():
                on = int(np.argmax(bo)); idx = [on - 2 * S, on - S, on, on + S, on + 2 * S]
                if idx[0] >= 0 and idx[-1] < T and bo[on:on + 2].all(): out.append([key, idx])
    return out


def _frames(clip):
    return [Image.fromarray(D.read_frames(_k2p[clip[0]], "camera_rs_0", [i])[0]) for i in clip[1]]


# ------------------------------------------------------- 1. VLM-labeled clip pairs
def collect_vlm_pairs(target=500, cap_usd=8.0, seed=7):
    flip, fall, safe = build_pools()
    onp = [w for w in safe if w["on_plate"]]
    pools = {"motion": ([w for w in safe if w["disp"] < .012], [w for w in safe if w["disp"] > .05]),
             "release": ([w for w in onp if w["tz"] < .27], [w for w in onp if w["tz"] > .35]),
             "position": ([w for w in onp if w["dc"] < .035], [w for w in onp if w["dc"] > .06]),
             "failure_mode": (flip, fall), "safety": (safe, flip + fall)}
    axes = ["safety", "failure_mode", "motion", "position", "release"]; wts = np.array([.22, .25, .30, .10, .13])
    rng = np.random.default_rng(seed)
    def w5(w): return [w["key"], w["idx"]] if isinstance(w, dict) else list(w)
    B = Budget(cap_usd, CACHE / "pref_clip_cost.json"); data = []
    while len(data) < target and B.calls < 2 * target:
        ax = rng.choice(axes, p=wts); PA, PB = pools[ax]
        if not PA or not PB: continue
        a, b = w5(PA[rng.integers(len(PA))]), w5(PB[rng.integers(len(PB))])
        if rng.random() < .5: a, b = b, a
        r = parse_json(B.call(MODEL, [PROMPT, "Clip A:"] + _frames(a) + ["Clip B:"] + _frames(b), max_tokens=110, thinking_budget=0)[0])
        if r and r.get("better") in ("A", "B"):
            data.append({"clipA": a, "clipB": b, "axis": ax, "better": r["better"], "deciding_rule": r.get("deciding_rule"),
                         "A_outcome": r.get("A_outcome"), "B_outcome": r.get("B_outcome")})
        if B.calls % 50 == 0: print(f"  calls {B.calls} kept {len(data)} ${B.spent:.3f}", flush=True)
    DATASET.write_text(json.dumps(data, indent=1)); shutil.copy(DATASET, CACHE / "pref_clip_dataset_vlmonly.json")
    print(f"SAVED {len(data)} VLM pairs (equal rate {1-len(data)/max(B.calls,1):.0%}) ${B.spent:.3f}")


# ------------------------------------------------------- 2 & 3. rubric-derived (no VLM) augmentations
def _mk(pref_pool, disp_pool, n, axis, out, rng):
    for _ in range(n):
        g, b = list(pref_pool[rng.integers(len(pref_pool))]), list(disp_pool[rng.integers(len(disp_pool))])
        better = "A" if rng.random() < .5 else "B"
        a, c = (g, b) if better == "A" else (b, g)
        out.append({"clipA": a, "clipB": c, "axis": axis, "better": better, "deciding_rule": "safety" if axis != "both" else "failure_mode"})


def add_both_pairs(n_safe=60, n_flip=45, n_fall=45, seed=11):
    """safe ≻ both (safety) and single-failure ≻ both (both is worst). Rubric-deterministic."""
    flip, fall, safe = build_pools(); both = both_clips()
    safe = [[w["key"], w["idx"]] for w in safe]; rng = np.random.default_rng(seed); new = []
    _mk(safe, both, n_safe, "safety", new, rng); _mk(flip, both, n_flip, "both", new, rng); _mk(fall, both, n_fall, "both", new, rng)
    _append(new, "both"); print(f"added {len(new)} both pairs")


def add_crosslevel_pairs(n_flip=90, n_fall=45, n_both=45, seed=23):
    """dispreferred-safe (fast/edge/high-release) ≻ each failure (R1 pin, weighted to flipped)."""
    flip, fall, safe = build_pools(); both = both_clips(); onp = [w for w in safe if w["on_plate"]]
    disp_safe = [[w["key"], w["idx"]] for w in ([w for w in safe if w["disp"] > .05] +
                 [w for w in onp if w["dc"] > .06] + [w for w in onp if w["tz"] > .35])]
    rng = np.random.default_rng(seed); new = []
    _mk(disp_safe, flip, n_flip, "crosslevel", new, rng); _mk(disp_safe, fall, n_fall, "crosslevel", new, rng); _mk(disp_safe, both, n_both, "crosslevel", new, rng)
    _append(new, "crosslevel"); print(f"added {len(new)} cross-level pairs")


def _append(new, tag):
    base = CACHE / "pref_clip_dataset_vlmonly.json"
    data = json.loads(DATASET.read_text())
    data = (json.loads(base.read_text()) if any(d.get("axis") == tag for d in data) and base.exists() else data) + new
    DATASET.write_text(json.dumps(data, indent=1))


def label_all():
    collect_vlm_pairs(); add_both_pairs(); add_crosslevel_pairs()


if __name__ == "__main__":
    label_all()

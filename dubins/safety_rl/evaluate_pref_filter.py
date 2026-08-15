#!/usr/bin/env python
"""Measure the goodness of the latent safety filter (preference setup).

For each eps we sample start states the LEARNED value deems recoverable
(V(start) > eps, i.e. NOT in the failure BRT / doomed set), roll out the least-
restrictive filter with POSTERIOR grounding, and report:
  * failure rate  -- fraction of trajectories that enter the ground-truth failure
                     set (inside obstacle OR off-sidewalk). Since starts are
                     recoverable, any failure is a genuine filter failure.
  * value stats   -- mean / min / max / std of the learned value V along each
                     trajectory, aggregated over trajectories.
  * extras        -- intervention rate, episode length, value/GT consistency.

Run from repo root, e.g.:
  python dubins/safety_rl/evaluate_pref_filter.py --profile set_b --eps 0.0,1.0,3.0 --n 100
"""
import sys, os, glob, json, argparse, pathlib
import numpy as np, torch, gym
import ruamel.yaml as yaml
from torch import nn
from torch.nn.utils import spectral_norm
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "dubins/safety_rl")
wm_root = pathlib.Path("dubins/dreamerv3").resolve()
for p in (str(wm_root), str(wm_root.parent)):
    if p not in sys.path: sys.path.insert(0, p)
from gym_reachability import gym_reachability
from RARL.model import Model
import tools, models
import generate_data_dubins_sidewalk as gen

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="set_b", choices=["set_a", "set_b"])
ap.add_argument("--eps", default="0.0", help="comma-separated eps thresholds to sweep")
ap.add_argument("--n", type=int, default=100, help="# trajectories per eps")
ap.add_argument("--max_steps", type=int, default=150)
ap.add_argument("--nominal", default="constant", choices=["constant", "random"])
ap.add_argument("--start_thresh", type=float, default=None,
                help="require V(start) > this (default: eps). Guarantees starts are not in the BRT.")
ap.add_argument("--start_xlim", type=float, default=1.0, help="sample starts with |x|<=this (interior, so they must traverse the obstacle region before exiting)")
ap.add_argument("--start_ylim", type=float, default=0.95, help="sample starts with |y|<=this")
ap.add_argument("--q_ckpt", default=None, help="explicit Q-*.pth (default: latest in the run dir)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="json output path")
a = ap.parse_args()
EPS_LIST = [float(e) for e in a.eps.split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(a.seed); torch.manual_seed(a.seed)

def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base: recursive_update(base[k], v)
        else: base[k] = v
_cfg = yaml.YAML(typ="safe", pure=True).load(pathlib.Path("dubins/configs.yaml").resolve().read_text())
defaults = {}; recursive_update(defaults, _cfg["defaults"])
class Config: pass
config = Config()
for k, v in defaults.items(): setattr(config, k, v)
config.device = device; config.num_actions = 3; config.use_margin_head = False

_CKPTS = {
    "set_a": dict(lx="checkpoints/pref_margin_head_set_a.pt",
                  q_dir="logs/experiments_pref/car-DDQN/pref_seta-toEnd/model"),
    "set_b": dict(lx="checkpoints/pref_margin_head_set_b.pt",
                  q_dir="logs/experiments_pref/car-DDQN/pref_setb-toEnd/model"),
}

# spaces + models
bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2*np.pi]])
low, high = bounds[:, 0], bounds[:, 1]; mid, ivl = (low+high)/2, high-low
isz = config.size[0]
obs_space = gym.spaces.Dict({
    "state": gym.spaces.Box(np.float32(mid-ivl/2), np.float32(mid+ivl/2)),
    "obs_state": gym.spaces.Box(-1, 1, (2,), np.float32),
    "image": gym.spaces.Box(0, 255, (isz, isz, 3), np.uint8)})
act_space = gym.spaces.Discrete(3)

wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
_ck = torch.load("checkpoints/wm_transition_only.pt", map_location=device)
_pfx = "_wm._orig_mod."
_sd = {k[len(_pfx):] if k.startswith(_pfx) else k: v for k, v in _ck["agent_state_dict"].items() if "_wm" in k}
wm.load_state_dict(_sd, strict=True); wm.eval(); wm.dynamics.sample = False

class MarginHead(nn.Module):
    def __init__(self, feat_size, units=512, layers=2, lipschitz=True):
        super().__init__()
        sn = spectral_norm if lipschitz else (lambda m: m)
        d, mods = feat_size, []
        for _ in range(layers): mods += [sn(nn.Linear(d, units)), nn.SiLU()]; d = units
        self.body = nn.Sequential(*mods); self.out = sn(nn.Linear(d, 1))
    def forward(self, feat): return self.out(self.body(feat)).squeeze(-1)
lx_mlp = MarginHead(config.dyn_stoch + config.dyn_deter, config.units).to(device)
lx_mlp.load_state_dict(torch.load(_CKPTS[a.profile]["lx"], map_location=device)); lx_mlp.eval()

env = gym.make("dubins_car_latent_avoid-v1", config=config, device=device,
               mode=config.mode, doneType=config.doneType, sample_inside_obs=True)
env.car.set_wm(wm, lx_mlp, config=config)
env.set_speed(speed=config.speed); env.set_constraint(radius=config.obs_r)
env.set_radius_rotation(R_turn=config.speed/config.turnRate); env.set_seed(a.seed)

stateDim = config.dyn_stoch + config.dyn_deter
actionNum = env.action_space.n; actionList = np.arange(actionNum)
Q_network = Model([stateDim] + list(config.architecture) + [actionNum], "Tanh").to(device)
if a.q_ckpt:
    Q_ckpt = a.q_ckpt
else:
    _qs = sorted(glob.glob(_CKPTS[a.profile]["q_dir"] + "/Q-*.pth"), key=lambda p: int(p.split("-")[-1][:-4]))
    assert _qs, f"no Q-*.pth in {_CKPTS[a.profile]['q_dir']}"
    Q_ckpt = _qs[-1]
Q_network.load_state_dict(torch.load(Q_ckpt, map_location=device)); Q_network.eval()
print(f"profile={a.profile}  Q={Q_ckpt}  nominal={a.nominal}  n={a.n}  max_steps={a.max_steps}", flush=True)

# ---- filter primitives (same logic as the rollout filter) ----
R, SW = config.obs_r, gen.SIDEWALK_Y
DOM = config.x_max
def gt_fail(state):                    # ground-truth failure set (profile-independent)
    x, y = float(state[0]), float(state[1])
    return (np.hypot(x, y) < R) or (abs(y) > SW)
def in_domain(state):
    return abs(float(state[0])) <= DOM and abs(float(state[1])) <= DOM
def q_values_of(feat):
    with torch.no_grad(): return Q_network(feat)[0]
def value_of(feat): return torch.max(q_values_of(feat)).item()
def feat_of(post): return env.car.wm.dynamics.get_feat(post)
def nominal():
    return 1 if a.nominal == "constant" else int(np.random.choice(actionList))
def posterior_step(post, action_idx, image, theta):
    prev = {k: (v[:, 0] if v.dim() > 1 and v.shape[1] == 1 else v) for k, v in post.items()}
    act = torch.zeros((1, 3), device=device); act[0, action_idx] = 1.0
    data = {"image": np.asarray(image, np.float32)[None, None],
            "obs_state": np.array([[[np.cos(theta), np.sin(theta)]]], np.float32),
            "action": act.detach().cpu().numpy()[None],
            "is_first": np.zeros((1, 1)), "is_terminal": np.zeros((1, 1))}
    data = env.car.wm.preprocess(data)
    with torch.no_grad():
        embed = env.car.wm.encoder(data)
        post, _ = env.car.wm.dynamics.obs_step(prev, act, embed[:, 0], torch.zeros(1, device=device), sample=False)
    return post

def sample_start(thresh, max_tries=4000):
    """A start NOT in the BRT: learned value V > thresh, not already failed, and in
    the interior box (|x|<=start_xlim, |y|<=start_ylim) so it must traverse the
    obstacle region rather than immediately exiting an edge."""
    for _ in range(max_tries):
        env.reset()
        post = {k: v[:, 0] for k, v in env.car.latent.items()}
        st = np.array(env.car.state, dtype=np.float32)
        if abs(float(st[0])) > a.start_xlim or abs(float(st[1])) > a.start_ylim:
            continue
        if (not gt_fail(st)) and value_of(feat_of(post)) > thresh:
            return post, st
    raise RuntimeError(f"could not sample a start with V > {thresh}")

def eval_rollout(eps, start_thresh):
    post, st = sample_start(start_thresh)
    feat = feat_of(post)
    values = [value_of(feat)]; failed = gt_fail(st); fail_step = -1; interventions = 0
    term = "timeout"
    for t in range(1, a.max_steps + 1):
        q = q_values_of(feat); a_nom = nominal()
        if q[a_nom].item() <= eps:            # least-restrictive safety filter
            action = int(q.argmax().item()); interventions += 1
        else:
            action = a_nom
        latent, _, done, _ = env.step(action)
        st = np.array(env.car.state, dtype=np.float32)
        post = posterior_step(post, action, latent["image"], float(st[2]))
        feat = feat_of(post); values.append(value_of(feat))
        if gt_fail(st):
            failed = True; fail_step = t; term = "FAIL"; break
        if not in_domain(st):
            term = "exit_domain"; break
        if done:
            term = "env_done"; break
    v = np.array(values, np.float32)
    return dict(failed=bool(failed), fail_step=fail_step, steps=len(v) - 1, term=term,
                interventions=interventions, intervene_rate=interventions / max(len(v) - 1, 1),
                v_min=float(v.min()), v_mean=float(v.mean()), v_max=float(v.max()),
                v_std=float(v.std()), v_start=float(v[0]), v_final=float(v[-1]))

def hist_mode(x, bins=25):
    if len(x) == 0: return float("nan")
    h, e = np.histogram(x, bins=bins); i = int(h.argmax()); return float(0.5 * (e[i] + e[i + 1]))

results = {}
for eps in EPS_LIST:
    thresh = a.start_thresh if a.start_thresh is not None else eps
    rolls = [eval_rollout(eps, thresh) for _ in range(a.n)]
    fr = np.mean([r["failed"] for r in rolls])
    vmins = np.array([r["v_min"] for r in rolls]); vmeans = np.array([r["v_mean"] for r in rolls])
    vmaxs = np.array([r["v_max"] for r in rolls]); steps = np.array([r["steps"] for r in rolls])
    interv = np.array([r["intervene_rate"] for r in rolls])
    failed_mask = np.array([r["failed"] for r in rolls])
    term_counts = {k: int(sum(r["term"] == k for r in rolls)) for k in ("FAIL", "exit_domain", "env_done", "timeout")}
    # value/GT consistency: of failures, how many did the value flag (min V <= eps) before failing?
    flagged = np.mean([r["v_min"] <= eps for r in rolls if r["failed"]]) if failed_mask.any() else float("nan")
    res = {
        "eps": eps, "start_thresh": thresh, "n": a.n, "failure_rate": float(fr),
        "traj_v_min":  {"mean": float(vmins.mean()), "min": float(vmins.min()), "max": float(vmins.max()),
                        "mode": hist_mode(vmins),
                        "frac_below_0": float(np.mean(vmins < 0)), "frac_below_eps": float(np.mean(vmins < eps))},
        "traj_v_mean": {"mean": float(vmeans.mean()), "min": float(vmeans.min()), "max": float(vmeans.max())},
        "traj_v_max":  {"mean": float(vmaxs.mean()),  "max": float(vmaxs.max())},
        "overall_v":   {"min": float(vmins.min()), "max": float(vmaxs.max()),
                        "mean": float(np.concatenate([[r["v_mean"]] for r in rolls]).mean())},
        "intervene_rate_mean": float(interv.mean()),
        "episode_len_mean": float(steps.mean()),
        "fail_step_mean": float(np.mean([r["fail_step"] for r in rolls if r["failed"]])) if failed_mask.any() else float("nan"),
        "failed_min_V_leq_eps_frac": float(flagged),
        "termination": term_counts,
    }
    results[f"eps={eps}"] = res
    print(f"\n===== eps={eps}  (starts: V>{thresh}, not in BRT;  {a.n} trajectories) =====")
    print(f"  FAILURE RATE            : {fr*100:5.1f}%   ({int(fr*a.n)}/{a.n} entered GT failure)")
    print(f"  value along trajectory (learned V):")
    print(f"    per-traj min-V : mean={vmins.mean():+6.2f}  min={vmins.min():+6.2f}  max={vmins.max():+6.2f}  "
          f"mode={hist_mode(vmins):+6.2f}  frac(<0)={np.mean(vmins<0)*100:4.0f}%  frac(<eps)={np.mean(vmins<eps)*100:4.0f}%")
    print(f"    per-traj mean  : mean={vmeans.mean():+6.2f}  min={vmeans.min():+6.2f}  max={vmeans.max():+6.2f}")
    print(f"    per-traj max   : mean={vmaxs.mean():+6.2f}  max={vmaxs.max():+6.2f}")
    print(f"    overall        : min={vmins.min():+6.2f}  max={vmaxs.max():+6.2f}")
    print(f"  intervention rate (mean): {interv.mean()*100:5.1f}%")
    print(f"  episode length (mean)   : {steps.mean():5.1f} steps"
          + (f"   (failed at mean step {res['fail_step_mean']:.0f})" if failed_mask.any() else ""))
    print(f"  termination             : FAIL={term_counts['FAIL']}  exit_domain={term_counts['exit_domain']}  "
          f"env_done={term_counts['env_done']}  timeout={term_counts['timeout']}")
    if failed_mask.any():
        print(f"  of failures, value flagged (min V<=eps): {flagged*100:.0f}%")

out = a.out or f"logs/experiments_pref/filter_eval_{a.profile}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"profile": a.profile, "q_ckpt": Q_ckpt, "nominal": a.nominal, "results": results},
          open(out, "w"), indent=2)
print(f"\nsaved -> {out}")

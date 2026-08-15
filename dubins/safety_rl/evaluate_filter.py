"""Numerical scorecard for the latent reach-avoid safety filter.

Rolls the least-restrictive filter (apply nominal if Q(s,a_nom) > eps, else the
argmax-Q action) from certified-safe starts and reports:

  [1] Safety outcome  : failure rate, step-violation rate (obstacle/sidewalk),
                        mean survival length.
  [2] eps sweep       : failure% vs intervention% vs mean-survival -- the
                        safety/least-restrictiveness trade-off curve.
  [4] Steering quality: action-advantage spread (relative to the value range),
                        argmax==one-step-lookahead-best agreement, and whether
                        the chosen action improves the (trusted) value.

(#3 -- classification vs a ground-truth sidewalk BRT -- is TODO; it needs the
avoid-BRT recomputed on the (x,y,theta) grid, since LS_BRT_v1_w1.25.npy is the
old center-only grid.)

Usage (from repo root):
    python dubins/safety_rl/evaluate_filter.py --profile set_a
    python dubins/safety_rl/evaluate_filter.py --profile set_b \
        --q_ckpt checkpoints/ddqn_set_b/model/Q-500001.pth \
        --n 30 --steps 45 --out filter_scorecard_setb.png
"""
import os, sys, glob, pathlib, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np, torch, gym
import ruamel.yaml as yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent          # dubins/safety_rl
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "dreamerv3-torch")); sys.path.insert(0, str(REPO))
import tools, models
import generate_data_dubins_sidewalk as gen
from gym_reachability import gym_reachability  # noqa
from RARL.model import Model

PROFILES = {
    "set_a": dict(wm="checkpoints/wm_margin_set_a.pt",
                  q_dir="checkpoints/ddqn_set_a/model"),
    "set_b": dict(wm="checkpoints/wm_margin_set_b.pt",
                  q_dir="checkpoints/ddqn_set_b/model"),
}


class _MH(torch.nn.Module):
    def __init__(s, h): super().__init__(); s.head = h
    def forward(s, f): return s.head(f).mode()


def load_config(device):
    cfg = yaml.YAML(typ="safe", pure=True).load((HERE.parent / "configs.yaml").read_text())["defaults"]
    config = argparse.Namespace(**{k: tools.args_type(v)(v) for k, v in cfg.items()})
    config.num_actions = 3; config.device = device
    return config


def build(profile, wm_ckpt, q_ckpt, config, device):
    env = gym.make("dubins_car_latent_avoid-v1", config=config, device=device,
                   mode=config.mode, doneType=config.doneType, sample_inside_obs=True)
    env.set_speed(config.speed); env.set_constraint(radius=config.obs_r)
    env.set_radius_rotation(R_turn=config.speed / config.turnRate); env.set_seed(0)
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(device)
    ck = torch.load(str(REPO / wm_ckpt), map_location=device); pre = "_wm._orig_mod."
    sd = {k[len(pre):] if k.startswith(pre) else k[14:]: v for k, v in ck["agent_state_dict"].items() if "_wm" in k}
    wm.load_state_dict(sd, strict=True); wm.eval(); wm.dynamics.sample = False
    env.car.set_wm(wm, _MH(wm.heads["margin"]).to(device), config)
    Q = Model([config.dyn_stoch + config.dyn_deter] + list(config.architecture) + [3], "Tanh").to(device)
    Q.load_state_dict(torch.load(str(REPO / q_ckpt) if not os.path.isabs(q_ckpt) else q_ckpt, map_location=device))
    Q.eval()
    return env, wm, Q


class Filter:
    def __init__(s, env, wm, Q, config, device):
        s.env, s.wm, s.Q, s.config, s.device = env, wm, Q, config, device
        s.controls = env.car.discrete_controls
        s.R, s.SW = config.obs_r, gen.SIDEWALK_Y

    def q(s, feat):
        with torch.no_grad(): return s.Q(feat)[0]

    def _embed(s, image, theta, a_idx, is_first):
        a = torch.zeros((1, 3), device=s.device); a[0, a_idx] = 1.0
        data = {"image": np.asarray(image, np.float32)[None, None],
                "obs_state": np.array([[[np.cos(theta), np.sin(theta)]]], np.float32),
                "action": a.detach().cpu().numpy()[None],
                "is_first": np.array([[is_first]], np.float32), "is_terminal": np.zeros((1, 1))}
        data = s.wm.preprocess(data)
        with torch.no_grad(): return s.wm.encoder(data), a

    def reset_to(s, state):
        s.env.reset()
        emb, a = s._embed(s.env.capture_image(np.array(state, np.float32)), state[2], 1, 1.0)
        with torch.no_grad():
            post, _ = s.wm.dynamics.observe(emb, a[None], torch.ones((1, 1), device=s.device))
        base = s.env.unwrapped
        base.state = np.array(state, np.float32); base.car.state = np.array(state, np.float32)
        return {k: v[:, 0] for k, v in post.items()}

    def step_post(s, post, a_idx, image, theta):
        prev = {k: (v[:, 0] if v.dim() > 1 and v.shape[1] == 1 else v) for k, v in post.items()}
        emb, a = s._embed(image, theta, a_idx, 0.0)
        with torch.no_grad():
            post, _ = s.wm.dynamics.obs_step(prev, a, emb[:, 0], torch.zeros(1, device=s.device), sample=False)
        return post

    def Vmax_at(s, state):                 # trusted value = max_a Q at a cold-posterior encoding of `state`
        img = s.env.capture_image(np.array(state, np.float32))
        with torch.no_grad():
            _, feat, _ = s.env.car.get_latent(np.array([state[0]]), np.array([state[1]]), np.array([state[2]]), [img])
            ft = feat if torch.is_tensor(feat) else torch.tensor(np.asarray(feat), dtype=torch.float32, device=s.device)
            ft = ft.to(s.device).float()
            if ft.dim() == 1: ft = ft[None]
            return float(s.Q(ft).max(1)[0].max().item())

    def g(s, st): return min(np.hypot(st[0], st[1]) - s.R, s.SW - abs(st[1]))

    def rollout(s, state0, eps, steps, lookahead=False):
        # rolls until the episode ends (car leaves the domain) or `steps` cap.
        post = s.reset_to(state0); st = np.array(state0, np.float32)
        rec = dict(failed=False, obs=0, sw=0, n=0, interv=0, survival=None, spreads=[], agree=[], improve=[])
        for t in range(steps):
            feat = s.wm.dynamics.get_feat(post); q = s.q(feat).detach().cpu().numpy()
            a_nom = 1; active = q[a_nom] <= eps; amax = int(q.argmax()); action = amax if active else a_nom
            rec["n"] += 1; rec["interv"] += int(active); rec["spreads"].append(float(q.max() - q.min()))
            if active and lookahead:
                Vcur = s.Vmax_at(st)
                Vn = [s.Vmax_at(s.env.car.integrate_forward(st, s.controls[a])) for a in range(3)]
                rec["agree"].append(int(amax == int(np.argmax(Vn))))
                rec["improve"].append(int(Vn[amax] >= Vcur - 1e-6))
            gv = s.g(st)
            if gv < 0:
                if not rec["failed"]: rec["survival"] = t
                rec["failed"] = True
                if np.hypot(st[0], st[1]) - s.R < 0: rec["obs"] += 1
                if s.SW - abs(st[1]) < 0: rec["sw"] += 1
            obs, cost, done, info = s.env.step(action)
            post = s.step_post(post, action, obs["image"], float(s.env.car.state[2]))
            st = np.array(s.env.car.state, np.float32)
            if done: break
        if rec["survival"] is None: rec["survival"] = rec["n"]   # never failed -> full episode length
        return rec


def value_range(F, ng=11):
    vals = []
    for th in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        for x in np.linspace(-1.5, 1.5, ng)[::2]:
            for y in np.linspace(-1.5, 1.5, ng)[::2]:
                vals.append(F.Vmax_at([x, y, th]))
    return np.array(vals)


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(device)
    p = PROFILES[a.profile]
    q_ckpt = a.q_ckpt or sorted(glob.glob(str(REPO / p["q_dir"]) + "/Q-*.pth"),
                                key=lambda x: int(x.split("-")[-1][:-4]))[-1]
    env, wm, Q = build(a.profile, a.wm_ckpt or p["wm"], q_ckpt, config, device)
    F = Filter(env, wm, Q, config, device)
    vr = value_range(F); vspan = float(vr.max() - vr.min())

    # fixed certified-safe starts (true margin > 0.3), reused across all eps for a fair sweep
    rng = np.random.default_rng(a.seed); starts = []
    while len(starts) < a.n:
        x, y = rng.uniform(-1.2, 1.2, 2); th = rng.uniform(0, 2 * np.pi)
        if F.g([x, y, th]) > 0.3: starts.append(np.array([x, y, th], np.float32))

    eps_list = [float(e) for e in a.eps_sweep.split(",")]
    hdr = (f"=== Filter metrics: {a.profile}  Q={pathlib.Path(q_ckpt).name}  "
           f"N={a.n} trajectories  cap={a.steps} steps ===\n"
           f"value(max_a Q) span {vspan:.2f}  range [{vr.min():.2f}, {vr.max():.2f}]\n"
           f"(#1 fail/viol/obs/sw/surv/interv, #4 advSpread/argmaxAgree/improveV)\n"
           f"{'eps':>5} {'fail%':>6} {'viol%':>6} {'obs':>5} {'sw':>5} {'surv':>6} {'interv%':>7} "
           f"{'advSpr%':>8} {'agree%':>7} {'imprV%':>7}")
    lines = [hdr]
    rows = []
    for eps in eps_list:
        recs = [F.rollout(s0, eps, a.steps, lookahead=True) for s0 in starts]
        nst = max(1, sum(r["n"] for r in recs))
        fail = np.mean([r["failed"] for r in recs]) * 100
        viol = sum(r["obs"] + r["sw"] for r in recs) / nst * 100
        obs = sum(r["obs"] for r in recs); sw = sum(r["sw"] for r in recs)
        surv = float(np.mean([r["survival"] for r in recs]))
        interv = sum(r["interv"] for r in recs) / nst * 100
        sp = np.concatenate([np.array(r["spreads"]) for r in recs]) if recs else np.array([0.0])
        ag = np.concatenate([np.array(r["agree"]) for r in recs if r["agree"]]) if any(r["agree"] for r in recs) else np.array([])
        im = np.concatenate([np.array(r["improve"]) for r in recs if r["improve"]]) if any(r["improve"] for r in recs) else np.array([])
        adv = sp.mean() / vspan * 100
        agp = ag.mean() * 100 if ag.size else float("nan")
        imp = im.mean() * 100 if im.size else float("nan")
        lines.append(f"{eps:>5.2f} {fail:>6.1f} {viol:>6.2f} {obs:>5d} {sw:>5d} {surv:>6.1f} {interv:>7.1f} "
                     f"{adv:>8.2f} {agp:>7.1f} {imp:>7.1f}")
        rows.append((eps, fail, interv))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        out = a.out
    else:
        out = str(REPO / f"logs/experiments/car-DDQN/filter_metrics_{a.profile}.txt")
    with open(out, "w") as fh:
        fh.write(txt + "\n")
    print(f"\nsaved metrics -> {out}")
    # eps-sweep trade-off plot
    plotp = out.rsplit(".", 1)[0] + ".png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "-o", label="failure %")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "-s", label="intervention %")
    ax.set_xlabel("eps (filter threshold)"); ax.set_ylabel("%")
    ax.set_title(f"{a.profile}  {pathlib.Path(q_ckpt).name}  (N={a.n})"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(plotp, dpi=130); print(f"saved plot -> {plotp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="set_a", choices=list(PROFILES))
    ap.add_argument("--q_ckpt", default=None, help="path to Q-*.pth (default: latest in the profile's v2 dir)")
    ap.add_argument("--wm_ckpt", default=None, help="override WM checkpoint")
    ap.add_argument("--n", type=int, default=100, help="# trajectories per eps")
    ap.add_argument("--steps", type=int, default=150, help="per-episode step cap (episodes end at out-of-bounds)")
    ap.add_argument("--eps_sweep", default="0.1,0.3,0.5,0.75")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the safe-start sampler")
    ap.add_argument("--out", default=None, help="metrics .txt path (plot saved alongside)")
    main(ap.parse_args())

"""Visualize the trained latent safety value function V(x) = max_a Q(x, a).

Left  : continuous safety value V(x) (seismic; >0 safe, <0 unsafe).
Right : discrete BRT -- the safe set {V > 0} vs the unsafe backward-reachable
        tube {V <= 0}.

Run AFTER the DDQN has produced a Q-*.pth checkpoint. From the repo root:

    python dubins/safety_rl/visualize_value_function.py            # latest Q ckpt
    python dubins/safety_rl/visualize_value_function.py \
        --q_ckpt checkpoints/ddqn_set_a/model/Q-500000.pth \
        --theta 1.5708 --n 81 --out value_function.png
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import gym
import ruamel.yaml as yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = pathlib.Path(__file__).resolve().parent          # dubins/safety_rl
REPO = HERE.parents[1]                                   # latent_pref_reachability
sys.path.insert(0, str(HERE))                            # RARL, gym_reachability
sys.path.insert(0, str(HERE.parent / "dreamerv3-torch"))  # models, tools
sys.path.insert(0, str(REPO))                            # generate_data_dubins_sidewalk
import tools
import models
import generate_data_dubins_sidewalk as gen
from gym_reachability import gym_reachability  # noqa: F401 (registers the env)
from RARL.model import Model


def load_config():
    cfg = yaml.YAML(typ="safe", pure=True).load(
        (HERE.parent / "configs.yaml").read_text())["defaults"]
    config = argparse.Namespace(**{k: tools.args_type(v)(v) for k, v in cfg.items()})
    config.num_actions = 3
    return config


def latest_q_ckpt():
    d = REPO / "checkpoints/ddqn_set_a/model"
    qs = sorted(d.glob("Q-*.pth"), key=lambda p: int(p.stem.split("-")[1]))
    if not qs:
        raise FileNotFoundError(f"no Q-*.pth found in {d}; train the DDQN first")
    return str(qs[-1])


class _MarginHead(torch.nn.Module):
    """wm.heads['margin'] (distribution head) -> predicted margin value."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feat):
        return self.head(feat).mode()


def build_env_and_wm(config, device):
    env = gym.make("dubins_car_latent_avoid-v1", config=config, device=device,
                   mode=config.mode, doneType=config.doneType, sample_inside_obs=True)
    env.set_speed(config.speed)
    env.set_constraint(radius=config.obs_r)
    env.set_radius_rotation(R_turn=config.speed / config.turnRate)
    env.set_seed(config.randomSeed)

    wm = models.WorldModel(env.observation_space, env.action_space, 0, config)
    ck = torch.load(str(REPO / config.wm_checkpoint), map_location=device)
    sd = {k[14:]: v for k, v in ck["agent_state_dict"].items() if "_wm" in k}
    wm.load_state_dict(sd)
    wm.dynamics.sample = False

    if getattr(config, "use_margin_head", False):
        lx = _MarginHead(wm.heads["margin"])
    else:
        lx, _ = wm._init_lx_mlp(config, 1)
        lx.load_state_dict(torch.load(
            str(REPO / config.lx_checkpoint), map_location=device)["agent_state_dict"])
    env.car.set_wm(wm, lx, config)
    return env


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    config.device = device

    env = build_env_and_wm(config, device)

    # --- Q-network ---
    state_dim = config.dyn_stoch + config.dyn_deter
    dim_list = [state_dim] + list(config.architecture) + [config.num_actions]
    Q = Model(dim_list, config.actType).to(device)
    q_ckpt = args.q_ckpt or latest_q_ckpt()
    Q.load_state_dict(torch.load(q_ckpt, map_location=device))
    Q.eval()
    print(f"loaded Q-network from {q_ckpt}")

    # --- headings matching the DDQN figure: 0, 90, 180, 270 degrees ---
    thetas = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    degs = [0, 90, 180, 270]
    nx = ny = args.n
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)
    ext = [env.bounds[0, 0], env.bounds[0, 1], env.bounds[1, 0], env.bounds[1, 1]]
    R, SW = config.obs_r, gen.SIDEWALK_Y

    grids = []
    for th in thetas:
        print(f"rendering + encoding {nx * ny} states at theta={th:.3f} ...")
        imgs, xl, yl, idx = [], [], [], []
        for i in range(nx):
            for j in range(ny):
                imgs.append(env.capture_image(np.array([xs[i], ys[j], th])))
                xl.append(xs[i]); yl.append(ys[j]); idx.append((i, j))
        xl, yl = np.array(xl), np.array(yl)
        with torch.no_grad():   # no grad graph -> far less GPU memory
            _, feat, _ = env.car.get_latent(xl, yl, np.full(len(xl), th), imgs)
            V = Q(torch.tensor(feat, dtype=torch.float32,
                               device=device)).max(dim=1)[0].cpu().numpy()
        g = np.zeros((nx, ny))
        for (i, j), v in zip(idx, V):
            g[i, j] = v
        grids.append(g)

    vmax = max(max(abs(g).max() for g in grids), 1e-3)

    def overlay(ax):
        ax.add_patch(Circle((0, 0), R, fill=False, color="k", lw=2, zorder=5))
        ax.axhspan(SW, env.bounds[1, 1], color="gray", alpha=0.3, zorder=4)
        ax.axhspan(env.bounds[1, 0], -SW, color="gray", alpha=0.3, zorder=4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")

    # rows: continuous value / discrete BRT ; columns: the four headings
    fig, axes = plt.subplots(2, len(thetas), figsize=(4 * len(thetas), 8))
    for c, (g, d) in enumerate(zip(grids, degs)):
        imc = axes[0, c].imshow(g.T, extent=ext, origin="lower", cmap="seismic",
                                vmin=-vmax, vmax=vmax, interpolation="none")
        axes[0, c].set_title(fr"$V(x)$   $\theta={d}^\circ$")
        overlay(axes[0, c])
        axes[1, c].imshow(g.T > 0, extent=ext, origin="lower", cmap="seismic",
                          vmin=-1, vmax=1, interpolation="none")
        axes[1, c].set_title(fr"BRT $\{{V>0\}}$   $\theta={d}^\circ$")
        overlay(axes[1, c])
    fig.colorbar(imc, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01)
    axes[0, 0].set_ylabel("continuous value", fontsize=13)
    axes[1, 0].set_ylabel("discrete (BRT)", fontsize=13)
    fig.suptitle(f"Learned latent safety value function  ({pathlib.Path(q_ckpt).name})",
                 fontsize=15)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q_ckpt", default=None,
                   help="path to a Q-*.pth (default: latest in the DDQN logdir)")
    p.add_argument("--n", type=int, default=51, help="grid resolution per axis")
    p.add_argument("--out", default="value_function.png")
    main(p.parse_args())

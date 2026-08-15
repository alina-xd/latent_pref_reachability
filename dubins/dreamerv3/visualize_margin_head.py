"""Visualize the jointly-trained margin HEAD (wm.heads['margin']) of a world
model checkpoint, using the EXACT same plot as dreamer_offline_sidewalk.py's
classifier eval (get_eval_plot) -- only the margin source differs.

We reuse Dreamer.get_eval_plot but pass a thin wrapper that returns the margin
head's prediction, so the head is a drop-in for the separate lx_mlp classifier.

Example:
    python visualize_margin_head.py \
        --wm_ckpt checkpoints/wm_margin_set_a.pt \
        --pref_profile set_a --out margin_head_eval.png
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
import gym
import ruamel.yaml as yaml
from torch import nn
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]     # latent_pref_reachability
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tools
import dreamer_offline_sidewalk as D   # Dreamer, PREF_PROFILES


class _DummyLogger:
    step = 0
    def scalar(self, *a): pass
    def write(self, *a, **k): pass
    def image(self, *a): pass
    def video(self, *a): pass


class MarginHeadAsClassifier(nn.Module):
    """Adapts wm.heads['margin'] (a distribution head) to the lx_mlp interface
    expected by get_latent: feat -> predicted margin value."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feat):
        return self.head(feat).mode()   # symexp of the symlog_mse mean


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.YAML(typ="safe", pure=True).load(
        (ROOT / "dubins" / "common" / "configs.yaml").read_text())["defaults"]
    cfg.update({"device": str(device), "compile": False, "use_margin_head": True,
                "x_min": -1.5, "x_max": 1.5, "y_min": -1.5, "y_max": 1.5})
    config = argparse.Namespace(**{k: tools.args_type(v)(v) for k, v in cfg.items()})
    config.num_actions = 3
    config.from_ckpt = None

    img = config.size[0]
    obs_space = gym.spaces.Dict({
        "state": gym.spaces.Box(np.float32([-1.5, -1.5, 0]), np.float32([1.5, 1.5, 6.3])),
        "obs_state": gym.spaces.Box(-1, 1, (2,), np.float32),
        "image": gym.spaces.Box(0, 255, (img, img, 3), np.uint8)})
    act_space = gym.spaces.Discrete(3)

    # Build the SAME agent used for classifier eval, then load the WM weights.
    agent = D.Dreamer(obs_space, act_space, config, _DummyLogger()).to(device)
    agent.requires_grad_(False)
    ck = torch.load(args.wm_ckpt, map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["agent_state_dict"].items()}
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    assert "margin" in agent._wm.heads, "checkpoint has no margin head"
    print(f"loaded WM (missing={len(missing)}, unexpected={len(unexpected)})")

    agent._pref_weights = D.PREF_PROFILES[args.pref_profile]
    agent.eval()

    # Exactly the classifier eval plot, but with the margin head as the source.
    lx_like = MarginHeadAsClassifier(agent._wm.heads["margin"]).to(device)
    plot, tp, fn, fp, tn = agent.get_eval_plot(lx_like)
    Image.fromarray(plot).save(args.out)
    fp_n, fn_n, tp_n, tn_n = fp[0].size, fn[0].size, tp[0].size, tn[0].size
    total = max(tp_n + tn_n + fp_n + fn_n, 1)
    print(f"margin head  sign-error={(fp_n + fn_n) / total:.3f}  -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # NOTE: pretrain_joint.pt was clobbered by a cancelled rerun; use the best_* file
    p.add_argument("--wm_ckpt", default="checkpoints/wm_margin_set_a.pt")
    p.add_argument("--pref_profile", default="set_a", choices=list(D.PREF_PROFILES))
    p.add_argument("--out", default="margin_head_eval.png")
    main(p.parse_args())

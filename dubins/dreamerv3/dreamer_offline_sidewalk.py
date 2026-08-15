import argparse
import collections
import io
import os
import pathlib
import sys
from io import BytesIO

os.environ["MUJOCO_GL"] = "osmesa"

import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml as yaml
import torch
from PIL import Image
from termcolor import cprint
from torch import nn
from tqdm import trange

sys.path.append(str(pathlib.Path(__file__).parent))
# repo root (…/latent_pref_reachability) for the dataset renderer / margin geometry
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import models
import tools
import generate_data_dubins_sidewalk as gen

to_np = lambda x: x.detach().cpu().numpy()


# ==================== Weighted signed-distance failure margin ====================
# Preference profiles for the latent failure classifier. Mirrors the privileged
# setup (Dubins_Privileged_Reachablity.ipynb): the margin is a signed distance
# to the nearest failure set (obstacle / sidewalk), scaled by a per-region
# preference weight. margin > 0 = safe, margin < 0 = unsafe.
PREF_PROFILES = {
    # avoid center strongly, prefer UPPER regions over lower
    "set_a": {"center": 3.0, "upper_sw": 1.0, "lower_sw": 1.0,
              "upper_safe": 3.0, "lower_safe": 0.5},
    # weight sidewalks heavily, prefer LOWER regions over upper, weak center
    "set_b": {"center": 1.0, "upper_sw": 3.0, "lower_sw": 3.0,
              "upper_safe": 1.0, "lower_safe": 3.0},
}


def weighted_margin_np(x, y, w, R, SW):
    """Scalar weighted signed-distance margin (numpy), matching the privileged
    setup's safety_margin()."""
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


def weighted_margin_torch(x, y, w, R, SW):
    """Vectorised weighted signed-distance margin (torch); same logic as
    weighted_margin_np, applied elementwise over (batch, time)."""
    l_center = torch.sqrt(x ** 2 + y ** 2) - R
    l_side = SW - torch.abs(y)
    base = torch.minimum(l_center, l_side)
    unsafe = base < 0
    in_obs = l_center < 0
    m = base
    m = torch.where(unsafe & in_obs, l_center * w["center"], m)
    m = torch.where(unsafe & (~in_obs) & (y > SW), l_side * w["upper_sw"], m)
    m = torch.where(unsafe & (~in_obs) & (y < -SW), l_side * w["lower_sw"], m)
    m = torch.where((~unsafe) & (y > 0), base * w["upper_safe"], m)
    m = torch.where((~unsafe) & (y <= 0), base * w["lower_safe"], m)
    return m


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        if config.compile and os.name != "nt":
            self._wm = torch.compile(self._wm)

        self._make_pretrain_opt()

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = config.precision == 16
        if config.pretrain_steps > 0 or config.from_ckpt is not None:
            # World model parameters: encoder + RSSM dynamics + decoder
            # Reward and cont heads are excluded (not used in offline WM pretraining)
            self.pretrain_params = (
                list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
                + list(self._wm.heads["decoder"].parameters())
            )
            # train the margin head jointly with the world model when enabled
            if "margin" in self._wm.heads:
                self.pretrain_params += list(self._wm.heads["margin"].parameters())
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt",
                self.pretrain_params,
                lr=config.model_lr,
                eps=config.opt_eps,
                clip=config.grad_clip,
                wd=config.weight_decay,
                opt=config.opt,
                use_amp=use_amp,
            )

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self):
        if self._logger is None:
            return
        if not self._should_log(self._step):
            return
        for name, values in self._metrics.items():
            if not np.isnan(np.mean(values)):
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []
        self._logger.write(fps=True)

    # ==================== World Model Pretraining ====================

    def pretrain_model_only(self, data, step):
        """Train world model (encoder + dynamics + decoder) with KL + reconstruction loss."""
        wm = self._wm
        data = wm.preprocess(data)

        with tools.RequiresGrad(wm):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: q(z_t | h_t, o_t), prior: p(z_t | h_t)
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior,
                    self._config.kl_free,
                    self._config.dyn_scale,
                    self._config.rep_scale,
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                if step <= self._config.pretrain_steps:
                    feat = wm.dynamics.get_feat(post)
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    # margin head: jointly regress the weighted signed-distance
                    # margin from the shared latent (gradients flow into the WM).
                    if "margin" in wm.heads:
                        R, SW = gen.OBS_R, gen.SIDEWALK_Y
                        x = data["privileged_state"][:, :, 0]
                        y = data["privileged_state"][:, :, 1]
                        margin_tgt = weighted_margin_torch(
                            x, y, self._pref_weights, R, SW).unsqueeze(-1)
                        margin_loss = -wm.heads["margin"](feat).log_prob(margin_tgt)
                        assert margin_loss.shape == embed.shape[:2], margin_loss.shape
                        losses["margin"] = margin_loss
                    # apply per-head loss scales (image/obs_state default to 1.0)
                    recon_loss = sum(
                        wm._scales.get(k, 1.0) * v for k, v in losses.items()
                    )
                else:
                    recon_loss = 0

                model_loss = kl_loss + recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )

        # Collect scalar metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )

        metrics = {f"model_only_pretrain/{k}": v for k, v in metrics.items()}
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
        self._logger.step = self._step

    # ==================== Observation Reconstruction Probe ====================

    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        """Train/evaluate an MLP to regress privileged state from prior latent features."""
        wm = self._wm
        data = wm.preprocess(data)

        if eval:
            obs_mlp.eval()

        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                # Use prior features to test how informative the learned prior is
                feat = wm.dynamics.get_feat(prior).detach()
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)

            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()

        return obs_loss.item()

    # ==================== Safety Classifier l(x) ====================

    def get_latent(self, xs, ys, thetas, imgs, lx_mlp):
        """Encode observations through WM and compute safety classifier output."""
        batch_size = xs.shape[0]
        states = np.expand_dims(np.expand_dims(thetas, 1), 1)
        imgs = np.expand_dims(imgs, 1)

        # Neutral action (index 1 = zero steering)
        dummy_acs = np.zeros((batch_size, 1, 3))
        dummy_acs[:, :, 1] = 1

        cos = np.cos(states)
        sin = np.sin(states)
        obs_state = np.concatenate([cos, sin], axis=-1)

        data = {
            "obs_state": obs_state,
            "image": imgs,
            "action": dummy_acs,
            "is_first": np.ones((batch_size, 1)),
            "is_terminal": np.zeros((batch_size, 1)),
        }
        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)
        post, _ = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():
            g_x = lx_mlp(feat).cpu().numpy().squeeze()
        feat_np = feat.cpu().numpy().squeeze()
        return g_x, feat_np, post

    def capture_image(self, state):
        """Render a synthetic observation for state [x, y, theta] using the SAME
        renderer as the training data (gray sidewalks + red obstacle + tree +
        top-down car icon), so eval-grid images match the WM's training
        distribution instead of the old outline+arrow rendering."""
        if not hasattr(self, "_emoji"):
            self._emoji = gen.load_tree_emoji()
        config = self._config
        return gen.render_state(
            np.asarray(state, dtype=np.float32),
            config.speed, config.dt, self._emoji, config.size[0],
        )

    def get_eval_plot(self, lx_mlp):
        """Evaluate the latent margin regressor over a grid: compare predicted
        l(x) against the ground-truth weighted signed-distance margin."""
        config = self._config
        R, SW = gen.OBS_R, gen.SIDEWALK_Y
        w = self._pref_weights

        nx, ny, nz = 41, 41, 5
        # cover the FULL domain so the sidewalk failure region is evaluated
        xs = np.linspace(config.x_min, config.x_max, nx)
        ys = np.linspace(config.y_min, config.y_max, ny)
        thetas = np.linspace(0, 2 * np.pi, nz, endpoint=True)

        pred = np.zeros((nx, ny, nz))   # predicted latent margin l(x)
        gt = np.zeros((nx, ny, nz))     # ground-truth weighted margin
        idxs, imgs = [], []
        it = np.nditer(pred, flags=["multi_index"])
        while not it.finished:
            i0, i1, i2 = it.multi_index
            x, y, theta = xs[i0], ys[i1], thetas[i2]
            gt[i0, i1, i2] = weighted_margin_np(x, y, w, R, SW)
            # render the actual grid state (image matches the labelled state)
            imgs.append(self.capture_image(np.array([x, y, theta])))
            idxs.append((i0, i1, i2))
            it.iternext()
        idxs = np.array(idxs)
        x_lin, y_lin, th_lin = xs[idxs[:, 0]], ys[idxs[:, 1]], thetas[idxs[:, 2]]

        # Predicted margin in chunks (memory)
        g_x, num_chunks = [], 10
        cs = int(np.ceil(len(x_lin) / num_chunks))
        for s in range(0, len(x_lin), cs):
            e = min(s + cs, len(x_lin))
            g_xk, _, _ = self.get_latent(
                x_lin[s:e], y_lin[s:e], th_lin[s:e], imgs[s:e], lx_mlp
            )
            g_x.extend(np.atleast_1d(g_xk).tolist())
        g_x = np.array(g_x)
        pred[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x

        # Sign-agreement confusion (safe = margin > 0) for a scalar score
        gt_flat = gt[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        safe_idxs = np.where(gt_flat > 0)
        unsafe_idxs = np.where(gt_flat <= 0)
        tp = np.where(g_x[safe_idxs] > 0)
        fn = np.where(g_x[safe_idxs] <= 0)
        fp = np.where(g_x[unsafe_idxs] > 0)
        tn = np.where(g_x[unsafe_idxs] <= 0)
        mse = float(np.mean((g_x - gt_flat) ** 2))

        # Plot: predicted l(x) | ground-truth l(x) | sign agreement, per theta
        extent = [config.x_min, config.x_max, config.y_min, config.y_max]
        vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1e-3)
        vmin = -vmax

        def _overlay(ax):
            ax.add_patch(plt.Circle((0, 0), R, fill=False, color="k", lw=1.5))
            ax.axhline(SW, color="k", ls="--", lw=1)
            ax.axhline(-SW, color="k", ls="--", lw=1)
            ax.set_aspect("equal")

        fig, axes = plt.subplots(nz, 3, figsize=(15, nz * 5))
        for i in range(nz):
            for col, field, title in [
                (0, pred[:, :, i].T, "predicted l(x)"),
                (1, gt[:, :, i].T, "ground-truth l(x)"),
            ]:
                ax = axes[i, col]
                im = ax.imshow(field, extent=extent, origin="lower",
                               cmap="seismic", vmin=vmin, vmax=vmax,
                               interpolation="none")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{title}  θ={thetas[i]:.1f}")
                _overlay(ax)
            ax = axes[i, 2]
            agree = (np.sign(pred[:, :, i].T) == np.sign(gt[:, :, i].T))
            ax.imshow(agree, extent=extent, origin="lower", cmap="RdYlGn",
                      vmin=0, vmax=1, interpolation="none")
            ax.set_title("sign(pred)==sign(gt)")
            _overlay(ax)

        tp_n, fn_n, fp_n, tn_n = tp[0].size, fn[0].size, fp[0].size, tn[0].size
        total = max(tp_n + fn_n + fp_n + tn_n, 1)
        fig.suptitle(
            f"MSE={mse:.3f}  TP={tp_n/total*100:.0f}% TN={tn_n/total*100:.0f}% "
            f"FP={fp_n/total*100:.0f}% FN={fn_n/total*100:.0f}%",
            fontsize=14,
        )
        fig.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB")), tp, fn, fp, tn

    def train_lx(self, data, lx_mlp, lx_opt, eval=False):
        """Train or evaluate the safety classifier on frozen world model features."""
        wm = self._wm
        wm.dynamics.sample = False

        if eval:
            lx_mlp.eval()
            plot_arr, tp, fn, fp, tn = self.get_eval_plot(lx_mlp)
            lx_mlp.train()

            fp_n, fn_n = np.shape(fp)[1], np.shape(fn)[1]
            tp_n, tn_n = np.shape(tp)[1], np.shape(tn)[1]
            print(f"TP: {tp_n}, FN: {fn_n}, TN: {tn_n}, FP: {fp_n}")
            score = (fp_n + fn_n) / (fp_n + fn_n + tp_n + tn_n)
            return score, plot_arr

        # Training: regress the weighted signed-distance margin from frozen
        # WM features (same target as the privileged setup, latent inputs).
        data = wm.preprocess(data)
        R, SW = gen.OBS_R, gen.SIDEWALK_Y
        w = self._pref_weights

        with tools.RequiresGrad(lx_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                feat = wm.dynamics.get_feat(post).detach()

                x = data["privileged_state"][:, :, 0]
                y = data["privileged_state"][:, :, 1]
                target = weighted_margin_torch(x, y, w, R, SW)   # (B, T)
                pred = lx_mlp(feat).squeeze(-1)                  # (B, T)

                lx_loss = torch.mean((pred - target) ** 2)
                lx_opt(lx_loss, lx_mlp.parameters())

        return lx_loss.item(), None


# ==================== Utilities ====================


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    return tools.from_generator(generator, config.batch_size)


# ==================== Main ====================


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    # classifier-only runs (pretrain_steps <= 0): skip WM pretraining and load a
    # frozen WM checkpoint. Disable torch.compile so checkpoint keys load cleanly.
    if not hasattr(config, "from_ckpt"):
        config.from_ckpt = None
    if config.pretrain_steps <= 0:
        config.compile = False

    # --- Directories ---
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    # --- Observation / action spaces ---
    action_space = gym.spaces.Discrete(3)
    config.num_actions = action_space.n

    bounds = np.array([
        [config.x_min, config.x_max],
        [config.y_min, config.y_max],
        [0, 2 * np.pi],
    ])
    low, high = bounds[:, 0], bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    image_size = config.size[0]

    observation_space = gym.spaces.Dict({
        "state": gym.spaces.Box(
            np.float32(midpoint - interval / 2),
            np.float32(midpoint + interval / 2),
        ),
        "obs_state": gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32,
        ),
        "image": gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8,
        ),
    })

    # --- Datasets ---
    expert_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)

    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_val_eps, config)

    print(f"Train episodes: {len(expert_eps)}, Val episodes: {len(expert_val_eps)}")

    # --- Agent ---
    agent = Dreamer(
        observation_space, action_space, config, logger,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # preference weights for the weighted signed-distance margin
    agent._pref_weights = PREF_PROFILES[getattr(config, "pref_profile", "set_a")]
    print(f"Preference profile: {getattr(config, 'pref_profile', 'set_a')} "
          f"-> {agent._pref_weights}")

    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])

    # --- Logging helpers ---
    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        logger.image("pretrain/" + title, np.transpose(np.array(plot), (2, 0, 1)))

    def eval_obs_recon():
        """Probe: train a small MLP to regress privileged state from prior features."""
        recon_steps = 101
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
        train_losses, eval_losses = [], []

        for i in range(recon_steps):
            if i % (recon_steps // 4) == 0:
                loss = agent.pretrain_regress_obs(
                    next(eval_dataset), obs_mlp, obs_opt, eval=True
                )
                eval_losses.append(loss)
            else:
                loss = agent.pretrain_regress_obs(
                    next(expert_dataset), obs_mlp, obs_opt
                )
                train_losses.append(loss)

        log_plot("train_recon_loss", train_losses)
        log_plot("eval_recon_loss", eval_losses)
        logger.scalar("pretrain/train_recon_loss_min", np.min(train_losses))
        logger.scalar("pretrain/eval_recon_loss_min", np.min(eval_losses))
        logger.write(step=logger.step)
        del obs_mlp, obs_opt
        return np.min(eval_losses)

    def evaluate():
        """Evaluate world model: video predictions + observation reconstruction probe."""
        agent.eval()

        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))
            video_pred = agent._wm.video_pred(next(expert_dataset))
            logger.video("train_recon/openl_agent", to_np(video_pred))

        logger.write(step=logger.step)
        recon_eval = eval_obs_recon()
        agent.train()
        return recon_eval

    def train_classifier():
        """Train the latent margin regressor l(x) on frozen world model features."""
        num_steps = 2501
        eval_interval = 250
        best_score = float("inf")
        tag = f"classifier_{getattr(config, 'pref_profile', 'set_a')}"
        lx_mlp, lx_opt = agent._wm._init_lx_mlp(config, 1)
        train_losses, eval_losses = [], []

        for i in range(num_steps):
            if i % eval_interval == 0:
                print("eval")
                score, eval_plot = agent.train_lx(
                    next(eval_dataset), lx_mlp, lx_opt, eval=True
                )
                eval_losses.append(score)
                logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
                logger.write(step=i + 100000)
                Image.fromarray(eval_plot).save(logdir / f"{tag}_eval_{i:05d}.png")
                best_score = tools.save_checkpoint(
                    tag, i, score, best_score, lx_mlp, logdir
                )
            else:
                loss, _ = agent.train_lx(
                    next(expert_dataset), lx_mlp, lx_opt
                )
                train_losses.append(loss)

        log_plot("train_lx_loss", train_losses)
        log_plot("eval_lx_loss", eval_losses)
        logger.scalar("pretrain/train_lx_loss_min", np.min(train_losses))
        logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_losses))
        logger.write(step=num_steps)
        print(f"Eval losses: {eval_losses}")
        return lx_mlp, lx_opt

    # ==================== Phase 1: World Model Pretraining ====================
    total_pretrain_steps = config.pretrain_steps
    if total_pretrain_steps > 0:
        cprint(
            f"Pretraining world model for {total_pretrain_steps} steps",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = (
            lambda step: "pretrain_joint"
            if step < config.pretrain_steps
            else "pretrain_actor"
        )
        best_score = float("inf")

        for step in trange(
            total_pretrain_steps,
            desc="World model pretraining",
            ncols=0,
            leave=False,
        ):
            if ((step + 1) % config.eval_every == 0) or step == 1:
                print("eval")
                score = evaluate()
                best_score = tools.save_checkpoint(
                    ckpt_name, step, score, best_score, agent, logdir
                )

            agent.pretrain_model_only(next(expert_dataset), step)

    # ==================== Phase 2: Safety Classifier Training ====================
    wm_ckpt = getattr(config, "wm_ckpt", None)
    wm_ckpt = pathlib.Path(wm_ckpt) if wm_ckpt else (logdir / "pretrain_joint.pt")
    print(f"Loading frozen WM checkpoint from {wm_ckpt}")
    checkpoint = torch.load(wm_ckpt, map_location=config.device)
    sd = checkpoint["agent_state_dict"]
    # align torch.compile key prefixes between checkpoint and this agent
    if not any("_orig_mod" in k for k in agent.state_dict().keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    print(f"  loaded WM (missing={len(missing)}, unexpected={len(unexpected)})")

    print(f"Training latent margin regressor l(x) [profile="
          f"{getattr(config, 'pref_profile', 'set_a')}]")
    lx_mlp, lx_opt = train_classifier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--pref_profile", default="set_a", choices=list(PREF_PROFILES))
    parser.add_argument("--wm_ckpt", default=None,
                        help="frozen WM checkpoint to load for classifier training")
    args, remaining = parser.parse_known_args()

    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    final_config = parser.parse_args(remaining)
    final_config.pref_profile = args.pref_profile
    final_config.wm_ckpt = args.wm_ckpt
    main(final_config)

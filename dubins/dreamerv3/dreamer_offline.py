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

import models
import tools

to_np = lambda x: x.detach().cpu().numpy()


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
                    recon_loss = sum(losses.values())
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
        """Render a synthetic observation image for a Dubins car state [x, y, theta]."""
        config = self._config
        fig, ax = plt.subplots()
        plt.xlim([config.x_min, config.x_max])
        plt.ylim([config.y_min, config.y_max])
        plt.axis("off")
        fig.set_size_inches(1, 1)

        # Obstacle boundary
        circle = patches.Circle(
            [0, 0], config.obs_r, edgecolor=(1, 0, 0), facecolor="none"
        )
        ax.add_patch(circle)

        # Agent heading arrow
        dt, v = config.dt, config.speed
        plt.quiver(
            state[0], state[1],
            dt * v * np.cos(state[2]), dt * v * np.sin(state[2]),
            angles="xy", scale_units="xy", minlength=0,
            width=0.05, scale=0.2, color=(0, 0, 1), zorder=3,
        )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=config.size[0])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img_array = np.array(img)
        plt.close()
        return img_array

    def get_eval_plot(self, lx_mlp):
        """Evaluate safety classifier over a grid and generate confusion matrix plot."""
        config = self._config
        nx, ny, nz = 41, 41, 5

        xs = np.linspace(-1, 1, nx)
        ys = np.linspace(-1, 1, ny)
        thetas = np.linspace(0, 2 * np.pi, nz, endpoint=True)

        # Build evaluation grid
        v = np.zeros((nx, ny, nz))
        idxs, imgs, labels = [], [], []
        it = np.nditer(v, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index
            x, y, theta = xs[idx[0]], ys[idx[1]], thetas[idx[2]]

            is_unsafe = (x ** 2 + y ** 2) < (config.obs_r ** 2)
            labels.append(1 if is_unsafe else 0)

            # Offset position one step backward along heading
            x_offset = x - np.cos(theta) * 0.05
            y_offset = y - np.sin(theta) * 0.05
            imgs.append(self.capture_image(np.array([x_offset, y_offset, theta])))
            idxs.append(idx)
            it.iternext()

        idxs = np.array(idxs)
        labels = np.array(labels)
        safe_idxs = np.where(labels == 0)
        unsafe_idxs = np.where(labels == 1)

        x_lin = xs[idxs[:, 0]]
        y_lin = ys[idxs[:, 1]]
        theta_lin = thetas[idxs[:, 2]]

        # Forward pass in chunks (memory constraint with large images)
        g_x = []
        num_chunks = 5
        chunk_size = len(x_lin) // num_chunks
        for k in range(num_chunks):
            s, e = k * chunk_size, (k + 1) * chunk_size
            g_xk, _, _ = self.get_latent(
                x_lin[s:e], y_lin[s:e], theta_lin[s:e], imgs[s:e], lx_mlp
            )
            g_x.extend(g_xk.tolist())
        g_x = np.array(g_x)
        v[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x

        # Confusion matrix counts
        tp = np.where(g_x[safe_idxs] > 0)
        fn = np.where(g_x[safe_idxs] <= 0)
        fp = np.where(g_x[unsafe_idxs] > 0)
        tn = np.where(g_x[unsafe_idxs] <= 0)

        # Plot
        vmax = round(max(np.max(v), 0), 1)
        vmin = round(min(np.min(v), -vmax), 1)
        extent = [config.x_min, config.x_max, config.y_min, config.y_max]

        fig, axes = plt.subplots(nz, 2, figsize=(12, nz * 6))
        for i in range(nz):
            # Left column: continuous g(x) values
            ax = axes[i, 0]
            im = ax.imshow(
                v[:, :, i].T, interpolation="none", extent=extent,
                origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1,
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=0.95,
                ticks=[vmin, 0, vmax],
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r"$g(x)$", fontsize=18)
            circle = plt.Circle(
                (0, 0), config.obs_r, fill=False, color="blue", label="GT boundary"
            )
            ax.add_patch(circle)
            ax.set_aspect("equal")

            # Right column: binary classification (g(x) > 0)
            ax = axes[i, 1]
            im = ax.imshow(
                v[:, :, i].T > 0, interpolation="none", extent=extent,
                origin="lower", cmap="seismic", vmin=-1, vmax=1, zorder=-1,
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=0.95,
                ticks=[vmin, 0, vmax],
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r"$v(x)$", fontsize=18)
            circle = plt.Circle(
                (0, 0), config.obs_r, fill=False, color="blue", label="GT boundary"
            )
            ax.add_patch(circle)
            ax.set_aspect("equal")

        fig.tight_layout()
        tp_n, fn_n = np.shape(tp)[1], np.shape(fn)[1]
        fp_n, tn_n = np.shape(fp)[1], np.shape(tn)[1]
        total = tp_n + fn_n + fp_n + tn_n
        fig.suptitle(
            r"$TP={:.0f}\%$ ".format(tp_n / total * 100)
            + r"$TN={:.0f}\%$ ".format(tn_n / total * 100)
            + r"$FP={:.0f}\%$ ".format(fp_n / total * 100)
            + r"$FN={:.0f}\%$".format(fn_n / total * 100),
            fontsize=10,
        )

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        return np.array(plot), tp, fn, fp, tn

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

        # Training: hinge loss on frozen WM features
        data = wm.preprocess(data)
        R = self._config.obs_r
        gamma = self._config.gamma_lx

        with tools.RequiresGrad(lx_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                feat = wm.dynamics.get_feat(post).detach()

                x = data["privileged_state"][:, :, 0]
                y = data["privileged_state"][:, :, 1]
                safety_margin = (x ** 2 + y ** 2) - R ** 2

                safe_feats = feat[safety_margin > 0]
                unsafe_feats = feat[safety_margin <= 0]

                pos = lx_mlp(safe_feats)    # should be positive (safe)
                neg = lx_mlp(unsafe_feats)  # should be negative (unsafe)

                # Hinge loss: penalize safe predictions below gamma, unsafe above -gamma
                lx_loss = (
                    torch.sum(torch.relu(gamma - pos)) / pos.size(0)
                    + torch.sum(torch.relu(gamma + neg)) / neg.size(0)
                )

                lx_opt(torch.mean(lx_loss), lx_mlp.parameters())

        return 0, None


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
        """Train safety classifier l(x) on frozen world model features."""
        num_steps = 2501
        eval_interval = 250
        best_score = float("inf")
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
                best_score = tools.save_checkpoint(
                    "classifier", i, score, best_score, lx_mlp, logdir
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
    print(f"Loading best WM checkpoint from {logdir / 'pretrain_joint.pt'}")
    checkpoint = torch.load(logdir / "pretrain_joint.pt")
    agent.load_state_dict(checkpoint["agent_state_dict"])

    print("Training safety classifier l(x)")
    lx_mlp, lx_opt = train_classifier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
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

    main(parser.parse_args(remaining))

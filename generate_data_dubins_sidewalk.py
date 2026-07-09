"""Generate world-model training data for the Dubins-car "sidewalk" scenario.

Scenario (see the setup figure):
  - Square domain of half-width / half-height 1.5  ->  x, y in [-1.5, 1.5].
  - A red circular obstacle of radius 0.5 centred at the origin.
  - Two "sidewalk" failure strips of height 0.3 at the top and bottom of the
    domain, i.e. |y| > 1.5 - 0.3 = 1.2.  These are drawn green.
  - A tree emoji is composited inside the red obstacle to give the world model
    a richer visual pattern to reason about.

The world model wants single-step state transitions grouped together, so we
roll out random-action trajectories, record every (state, action, next-state)
transition, render each state to an RGB observation, and dump everything to a
pickle file with the same schema that eais_hw2/dreamerv3-torch consumes:

    demo = {
      'obs':     {'image': [...], 'state': [theta...], 'priv_state': [[x,y,theta]...]},
      'actions': [scalar action in {-u_max, 0, u_max} ...],
      'dones':   [1, 1, ...],
    }

The car-rendering style (blue quiver + marker) mirrors
eais_hw2/generate_data_traj.py; the environment / obstacle / sidewalk drawing
is written specifically for this scenario.
"""

import argparse
import io
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------
HALF_W = 1.5          # domain half-width   -> x in [-1.5, 1.5]
HALF_H = 1.5          # domain half-height  -> y in [-1.5, 1.5]
OBS_R = 0.5           # radius of the central (red) obstacle
SIDEWALK_H = 0.4      # height of each (gray) sidewalk failure strip
SIDEWALK_Y = HALF_H - SIDEWALK_H  # |y| > SIDEWALK_Y is a sidewalk failure

# colours
SIDEWALK_COLOR = (0.55, 0.55, 0.55)  # gray sidewalk failure strips
RED = (0.85, 0.15, 0.15)

# top-down car icon colours
CAR_BODY = (0.11, 0.52, 0.93)   # bright blue body / roof
CAR_GLASS = (0.13, 0.13, 0.16)  # dark windows / sunroof
CAR_TAIL = (0.90, 0.10, 0.10)   # red rear tail-lights

NOTO_EMOJI_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
TREE_EMOJI = "\U0001F333"  # 🌳


# ---------------------------------------------------------------------------
# Tree emoji handling (matplotlib cannot rasterise the colour-bitmap font, so
# we render it once with PIL and composite it onto every observation).
# ---------------------------------------------------------------------------
def load_tree_emoji():
    """Return a tightly-cropped RGBA image of the tree emoji, or None."""
    try:
        # NotoColorEmoji only ships a single bitmap strike at size 109.
        font = ImageFont.truetype(NOTO_EMOJI_PATH, 109)
        canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((80, 80), TREE_EMOJI, font=font, embedded_color=True, anchor="mm")
        bbox = canvas.getbbox()
        if bbox is None:
            return None
        return canvas.crop(bbox)
    except Exception as exc:  # pragma: no cover - font/PIL fallbacks
        print(f"[warn] could not load tree emoji ({exc}); obstacle will be plain red")
        return None


def composite_tree(img, emoji, px):
    """Paste the tree emoji, scaled to `px` pixels, at the centre of `img`."""
    if emoji is None:
        return img
    px = max(1, int(px))
    tree = emoji.resize((px, px), Image.LANCZOS)
    W, H = img.size
    # obstacle centre is the origin -> centre of the (symmetric) image
    top_left = (W // 2 - px // 2, H // 2 - px // 2)
    img = img.convert("RGBA")
    img.alpha_composite(tree, top_left)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _to_world(pts, x, y, theta):
    """Rotate body-frame points by `theta` and translate to (x, y)."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(pts) @ rot.T + np.array([x, y])


def _rounded_rect(cx, cy, hl, hw, r_front, r_rear=None, n=8):
    """Rounded-rectangle outline centred at (cx, cy), +x toward the front.

    The front (+x) corners use `r_front`; the rear (-x) corners use `r_rear`
    (defaults to `r_front`), so the body can taper toward the nose.
    """
    if r_rear is None:
        r_rear = r_front
    r_front = min(r_front, hl, hw)
    r_rear = min(r_rear, hl, hw)
    corners = [
        (cx + hl - r_front, cy + hw - r_front, 0.0, np.pi / 2, r_front),        # front-left
        (cx - hl + r_rear, cy + hw - r_rear, np.pi / 2, np.pi, r_rear),         # rear-left
        (cx - hl + r_rear, cy - hw + r_rear, np.pi, 3 * np.pi / 2, r_rear),     # rear-right
        (cx + hl - r_front, cy - hw + r_front, 3 * np.pi / 2, 2 * np.pi, r_front),  # front-right
    ]
    pts = []
    for ccx, ccy, a0, a1, r in corners:
        ang = np.linspace(a0, a1, n)
        pts.append(np.column_stack([ccx + r * np.cos(ang), ccy + r * np.sin(ang)]))
    return np.vstack(pts)


def draw_car(ax, x, y, theta):
    """Draw a top-down sedan icon at (x, y) heading along `theta` (+x is front).

    Front cue: tapered blue nose + side mirrors.  Rear cue: red tail-lights.
    """
    def add(pts, color, z):
        ax.add_patch(patches.Polygon(
            _to_world(pts, x, y, theta), closed=True,
            facecolor=color, edgecolor="none", zorder=z))

    hl, hw = 0.20, 0.09  # body half-length / half-width

    # side mirrors (blue, just behind the nose, poking out sideways)
    for sign in (1, -1):
        add(_rounded_rect(0.085, sign * (hw + 0.017), 0.016, 0.013, 0.005), CAR_BODY, 3.0)

    # body: rounder/tapered front, boxier rear
    add(_rounded_rect(0.0, 0.0, hl, hw, r_front=0.085, r_rear=0.05), CAR_BODY, 3.1)

    # dark glass cabin (front + rear windshields and side windows)
    add(_rounded_rect(-0.015, 0.0, 0.15, 0.073, 0.045), CAR_GLASS, 3.2)

    # blue roof over the cabin, leaving the surrounding glass visible
    add(_rounded_rect(-0.025, 0.0, 0.10, 0.05, 0.03), CAR_BODY, 3.3)

    # dark sunroof in the centre of the roof
    add(_rounded_rect(-0.025, 0.0, 0.042, 0.03, 0.015), CAR_GLASS, 3.4)


def render_state(state, v, dt, emoji, dpi):
    """Render a single (x, y, theta) state to an RGB numpy observation."""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])

    fig, ax = plt.subplots()
    fig.set_size_inches(1, 1)
    ax.set_xlim([-HALF_W, HALF_W])
    ax.set_ylim([-HALF_H, HALF_H])
    ax.axis("off")

    # gray sidewalk failure strips (top and bottom)
    ax.add_patch(patches.Rectangle(
        (-HALF_W, SIDEWALK_Y), 2 * HALF_W, SIDEWALK_H,
        facecolor=SIDEWALK_COLOR, edgecolor="none", zorder=1))
    ax.add_patch(patches.Rectangle(
        (-HALF_W, -HALF_H), 2 * HALF_W, SIDEWALK_H,
        facecolor=SIDEWALK_COLOR, edgecolor="none", zorder=1))

    # red central obstacle (the tree emoji is composited on top in pixel space)
    ax.add_patch(patches.Circle(
        (0.0, 0.0), OBS_R, facecolor=RED, edgecolor="none", zorder=2))

    # car: top-down icon (body + windshields + wheels), front along theta
    draw_car(ax, x, y, theta)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # scale the emoji to ~80% of the obstacle diameter in pixels
    obstacle_diam_px = (2 * OBS_R) / (2 * HALF_W) * img.size[0]
    img = composite_tree(img, emoji, 0.8 * obstacle_diam_px)

    return np.array(img)


# ---------------------------------------------------------------------------
# Failure / sampling helpers
# ---------------------------------------------------------------------------
def off_domain(x, y):
    """True if the state is off-domain."""
    if abs(x) > HALF_W or abs(y) > HALF_H:  # left the domain
        return True
    return False


def sample_free_state():
    """Sample a random collision-free start state with a random heading."""
    margin = 0.05
    while True:
        x = np.random.uniform(-HALF_W + margin, HALF_W - margin)
        y = np.random.uniform(-(SIDEWALK_Y - margin), SIDEWALK_Y - margin)
        if not off_domain(x, y):
            break
    theta = np.random.uniform(0.0, 2 * np.pi)
    return torch.tensor([x, y, theta], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def gen_one_traj(u_max, dt, v, dpi):
    """Roll out one random-action trajectory and return its transitions."""
    emoji = load_tree_emoji()
    state = sample_free_state()
    action_choices = torch.tensor([-u_max, 0.0, u_max])

    state_obs, state_gt, img_obs, acs, dones = [], [], [], [], []

    for t in range(100):
        # stop the trajectory as soon as it is out of boundary
        if off_domain(float(state[0]), float(state[1])):
            break

        ac = action_choices[torch.randint(0, 3, (1,))].item()

        nxt = torch.empty(3)
        nxt[0] = state[0] + v * dt * torch.cos(state[2])
        nxt[1] = state[1] + v * dt * torch.sin(state[2])
        nxt[2] = state[2] + dt * ac

        # record this transition (each timestep is its own transition)
        state_obs.append(state[2].numpy())            # observed heading theta
        state_gt.append(state.numpy().copy())         # privileged full state
        acs.append(ac)
        dones.append(1)                               # is_last / is_terminal
        img_obs.append(render_state(state, v, dt, emoji, dpi))

        state = nxt

    return state_obs, acs, state_gt, img_obs, dones


def generate_trajs(num_pts, u_max, dt, v, dpi, out_path):
    demos = []
    num_samples = 3
    samples_saved = 0
    here = os.path.dirname(os.path.abspath(__file__))

    for i in range(num_pts):
        state_obs, acs, state_gt, img_obs, dones = gen_one_traj(
            u_max, dt, v, dpi)

        # skip degenerate (empty) rollouts
        if len(img_obs) == 0:
            continue

        # save a few sample observations (one frame from the first few rollouts)
        if samples_saved < num_samples:
            sample_path = os.path.join(
                here, f"sample_dubins_sidewalk_{samples_saved}.png")
            Image.fromarray(img_obs[0]).save(sample_path)
            print(f"saved sample observation -> {sample_path}")
            samples_saved += 1

        demos.append({
            "obs": {"image": img_obs, "state": state_obs, "priv_state": state_gt},
            "actions": acs,
            "dones": dones,
        })

        if (i + 1) % 50 == 0:
            print(f"generated {i + 1}/{num_pts} trajectories")

    with open(out_path, "wb") as f:
        pickle.dump(demos, f)
    print(f"saved {len(demos)} trajectories -> {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_pts", type=int, default=5000,
                        help="number of trajectories to roll out")
    parser.add_argument("--turnRate", type=float, default=1.25,
                        help="maximum turn rate u_max")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--time_limit", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=128,
                        help="output image size in pixels (dpi with a 1in figure)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"dubins_sidewalk_demos{args.dpi}.pkl")

    generate_trajs(args.num_pts, args.turnRate, args.dt, args.speed,
                   args.dpi, out_path)

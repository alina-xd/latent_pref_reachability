"""E0 - Load data.

Everything downstream reads from here:
  * raw dataset access (JPEG decode, per-frame failure labels, proprioception);
  * the DINO-WM feature store `egg/egg_wm.h5` (DINOv2 ViT-S/14-reg patch tokens),
    built by `build_features()` and read per-frame as a mean-pooled 384-d vector.

The stored JPEGs decode straight to RGB (orange yolk / blue plate), so `decode_frame`
does NOT apply a BGR->RGB swap. Features are extracted from `camera_zed_1`.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2, h5py, numpy as np

EGG = Path(__file__).resolve().parents[0]                  # egg/  (data lives here, self-contained)
REPO = Path(__file__).resolve().parents[1]                 # repo root (checkpoints + figures collected here)
DATA_ROOT = EGG / "data"                                   # raw trajectory hdf5s (read-only)
H5 = EGG / "egg_wm.h5"                                     # DINO-WM feature store
LABELS = EGG / "labels"                                    # VLM labels + clip-pair datasets
CKPT = REPO / "checkpoints" / "egg"                        # trained margins
FIG = REPO / "visualizations" / "egg"                      # figures
for _d in (LABELS, CKPT, FIG): _d.mkdir(parents=True, exist_ok=True)
CAMERAS = ("camera_rs_0", "camera_zed_1", "camera_zed_2")
FEATURE_CAMERA = "camera_zed_1"                            # what egg_wm.h5 cam_embd is extracted from

# per-frame class taxonomy (fallen, flipped) -> name
CLASS_OF = {(1, 1): "both", (1, 0): "fallen", (0, 1): "flipped", (0, 0): "safe"}


# --------------------------------------------------------------- raw dataset access
def list_trajectories(root: Path | str = DATA_ROOT) -> list[Path]:
    return sorted(Path(root).rglob("traj_*.hdf5"))


def decode_frame(buf: np.ndarray) -> np.ndarray:
    """JPEG bytes -> RGB uint8 (imdecode already yields RGB for this dataset)."""
    return cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def read_frames(path: Path, camera: str = FEATURE_CAMERA, idx=None) -> np.ndarray:
    with h5py.File(path, "r") as h:
        ds = h["data"][camera]
        idx = range(len(ds)) if idx is None else idx
        return np.stack([decode_frame(ds[int(i)]) for i in idx])


def read_lowdim(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        d = h["data"]
        ee = d["ee_states"][:].reshape(-1, 4, 4).transpose(0, 2, 1)   # stored column-major
        return {"actions": d["actions"][:], "ee_pos": ee[:, :3, 3], "ee_rot": ee[:, :3, :3],
                "gripper": d["gripper_states"][:], "joints": d["joint_states"][:],
                "joint_vel": d["joint_velocities"][:]}


@dataclass
class Traj:
    key: str; path: Path; n_steps: int
    fallen: np.ndarray; flipped: np.ndarray; fall_frame: int; flip_frame: int


def read_meta(path: Path) -> Traj:
    with h5py.File(path, "r") as h:
        fallen, flipped = h["labels/fallen"][:], h["labels/flipped"][:]; a = dict(h.attrs)
        return Traj(str(a.get("key", path.stem)), path, len(fallen), fallen, flipped,
                    int(a.get("fall_frame", -1)), int(a.get("flip_frame", -1)))


def frame_class(fallen: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    return np.array([CLASS_OF[(int(a), int(b))] for a, b in zip(fallen, flipped)])


# --------------------------------------------------------------- egg_wm.h5 feature store
def h5_key_to_path():
    """Map the h5 group attr 'key' -> source trajectory path (for reading raw frames)."""
    k2p = {f"{p.parent.name}/{p.stem}": p for p in list_trajectories()}
    with h5py.File(H5, "r") as hf:
        return {hf[hk].attrs["key"]: (hk, k2p[hf[hk].attrs["key"]])
                for hk in hf.keys() if hf[hk].attrs["key"] in k2p}


def frame_features(hf, hk, idx) -> np.ndarray:
    """mean-pooled (over 256 patches) DINO features for frames `idx` of group hk -> (len(idx),384)."""
    return np.stack([np.asarray(hf[hk]["cam_embd"][t], np.float32).mean(0) for t in idx])


def trajectory_features(hf, hk) -> np.ndarray:
    """(T,384) mean-pooled features for a whole trajectory."""
    return np.asarray(hf[hk]["cam_embd"], np.float32).mean(1)


# --------------------------------------------------------------- (re)build the feature store
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
_RES = 224                                                  # 224/14 = 16 -> 16x16 = 256 patches


def build_features(camera: str = FEATURE_CAMERA, out: Path = H5, batch: int = 128,
                   device: str = "cuda:0", limit: int = 0) -> None:
    """DINOv2 ViT-S/14-reg patch tokens for every frame -> consolidated h5.

    Group per trajectory: cam_embd (T,256,384) f16, labels (T,2) i8, attrs['key'].
    ~1500 fps; the whole dataset is ~2 min.
    """
    import torch, torch.nn.functional as Fnn
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").eval().to(device)
    mean, std = torch.tensor(_MEAN, device=device), torch.tensor(_STD, device=device)

    @torch.no_grad()
    def embed(imgs):
        x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float().div_(255)
        x = Fnn.interpolate(x, size=(_RES, _RES), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            return model.forward_features(x)["x_norm_patchtokens"].half().cpu().numpy()

    files = list_trajectories()[: limit or None]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as o:
        for i, path in enumerate(files):
            with h5py.File(path, "r") as h:
                ds = h["data"][camera]; T = len(ds); embs, buf = [], []
                for t in range(T):
                    buf.append(decode_frame(ds[t]))
                    if len(buf) == batch: embs.append(embed(np.stack(buf))); buf = []
                if buf: embs.append(embed(np.stack(buf)))
                labels = np.stack([h["labels/fallen"][:], h["labels/flipped"][:]], 1)
                key = str(dict(h.attrs).get("key", path.stem))
            g = o.create_group(f"trajectory_{i:04d}")
            g.create_dataset("cam_embd", data=np.concatenate(embs), dtype="float16")
            g.create_dataset("labels", data=labels.astype(np.int8))
            g.attrs["key"] = key
            if i % 50 == 0: print(f"[{i+1}/{len(files)}]", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    build_features()

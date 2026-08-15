# Preference-Aligned Latent Safety Filtering

Learning safety value functions whose values represent preferences, not just
safe/fail classification, but which failure mode and which safe behavior is preferred, from Bradley-Terry / regression on latent features, including VLM-generated labels. Validated in a Dubins car simulated toy environment and on a real egg-cooking robot arm scenario.

- 📄 **Writeup:** `https://drive.google.com/file/d/1gjRASsECn9K31yvaaTFAzDBM2xQewCui/view?usp=sharing`
- 🖼 **Poster:** `https://drive.google.com/file/d/1Czp43r9xdHzxmauoElLXvU0cHUC2hSfC/view?usp=sharing`

## Notebooks (run from the repo root)

| notebook | what it covers | code |
|---|---|---|
| [`01_Dubins_Privileged.ipynb`](01_Dubins_Privileged.ipynb) | privileged-state preference-aligned safety filter: env + reach-avoid value DDQN + filtering + rollout visualizations | `dubins/` |
| [`02_Dubins_Latent_GT.ipynb`](02_Dubins_Latent_GT.ipynb) | latent-state safety filter with a GT-label margin: world model + failure margin + latent safety filter | `dubins/` |
| [`03_Dubins_Latent_VLM.ipynb`](03_Dubins_Latent_VLM.ipynb) | full latent pipeline with VLM labels: VLM pairwise labels + BT margin + safety filter | `dubins/` |
| [`04_Egg.ipynb`](04_Egg.ipynb) | egg scenario: preference-aligned failure margins on DINOv2 features (BT vs regression, two preference sets) | `egg/` |

Each notebook opens with a markdown cell explaining what it does and embeds its key figures (so it reads correctly without rerunning).

## Virutal environment
A single conda env, `reach_pref`, runs all four notebooks. Select the `reach_pref` Jupyter kernel.
Gemini labeling (egg) requires your own Gemini API key. Place `KEY=<your_api_key>` in `~/.gemini_key`. The Dubins VLM labeler (Qwen3-VL) runs locally on GPU, no key needed.

## Layout
```
01_…04_*.ipynb        the four notebooks live at the root
dubins/               shared Dubins code (DreamerV3, safety_rl, the generate_* scripts)
egg/                  all egg code + egg data (self-contained, see egg/README.md)
```

## Getting the data & checkpoints

Large / derived / confidential datasets are git-ignored. To obtain:

### Dubins data — generatable with the scripts in this repo
```bash
# world-model training rollouts (~7.5 GB)
python dubins/generate_data_dubins_sidewalk.py --out data/wm_training_data.pkl
# preference labels, cautious (A) and speed (B) profiles
python dubins/generate_pref_labels_dubins_sidewalk.py --pref_profile set_a --out data/pref_labels_set_a.pkl
python dubins/generate_pref_labels_dubins_sidewalk.py --pref_profile set_b --out data/pref_labels_set_b.pkl
```

### Egg data — confidential
The raw egg trajectories are not uploaded here. When you obtain the dataset, put the trajectory `traj_*.hdf5` files under `egg/data/` to be read by `egg/load_data.py`.

### Checkpoints — generate by running the notebooks
All trained models are git-ignored. Egg: run `04_Egg.ipynb` with `RETRAIN=True`. Dubins: run the training cells in `02`/`03` (world model + DDQN, hours on a GPU).

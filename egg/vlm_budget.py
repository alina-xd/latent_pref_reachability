"""E1 - VLM budget / cost study.

Estimate the $ cost of a labeling run before launching it, and measure real per-call token usage.
Gemini bills per image token; the robotics-ER / 3.5-flash tiers tokenize one native frame at
~1064 tokens, the standard 2.5-flash tiers at ~275 -- the dominant cost lever for image labeling.
"""
from __future__ import annotations
from PIL import Image
import numpy as np
import load_data as D
from vlm_call import Budget, parse_json, PRICE

# measured input tokens per native egg frame (empirical, per model tier)
TOKENS_PER_IMAGE = {"gemini-robotics-er-1.6-preview": 1064, "gemini-3.5-flash": 1064,
                    "gemini-2.5-flash": 275, "gemini-2.5-flash-lite": 275}


def estimate(n_items, images_per_item, model="gemini-robotics-er-1.6-preview",
             out_tokens=60, prompt_tokens=350):
    """rough $ for n_items calls, each with `images_per_item` frames + a short prompt/answer."""
    pin, pout = PRICE[model]; tpi = TOKENS_PER_IMAGE.get(model, 1064)
    inp = n_items * (images_per_item * tpi + prompt_tokens)
    out = n_items * out_tokens
    usd = inp / 1e6 * pin + out / 1e6 * pout
    print(f"[{model}] {n_items} items x {images_per_item} img -> ~{inp/1e6:.2f}M in / {out/1e6:.2f}M out  =  ${usd:.2f}")
    return usd


def measure(model="gemini-robotics-er-1.6-preview", images_per_item=5):
    """one live call -> actual token usage, so the estimate above can be calibrated."""
    info = D.h5_key_to_path(); key = next(iter(info)); path = info[key][1]
    ims = [Image.fromarray(x) for x in D.read_frames(path, "camera_rs_0", list(range(0, 5 * images_per_item, 5)))]
    B = Budget(1.0)
    _, usage = B.call(model, ["How many images is this?"] + ims, max_tokens=20, thinking_budget=0)
    print(f"[{model}] live: prompt={usage.get('promptTokenCount')} tok  "
          f"(~{usage.get('promptTokenCount',0)/images_per_item:.0f}/image)  cost=${B.spent:.4f}")
    return usage


if __name__ == "__main__":
    estimate(500, 10)        # 500 clip pairs (2x5 frames)
    estimate(2000, 1)        # 2000 state frames

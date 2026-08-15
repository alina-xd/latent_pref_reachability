"""E1 - VLM caller.

Budget-tracked Gemini calls used by the labeling steps (label_states, label_pairs). Enforces a
hard USD cap and logs exact cost from each response's usageMetadata. Key is read from ~/.gemini_key.
Default model for this project: gemini-robotics-er-1.6-preview (thinkingBudget=0).
"""
from __future__ import annotations
import base64, io, json, time
from pathlib import Path
import requests
from PIL import Image

API = "https://generativelanguage.googleapis.com/v1beta/models"
PRICE = {  # (input, output) USD per 1e6 tokens
    "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.50, 3.00), "gemini-3.5-flash": (1.50, 9.00),
    "gemini-robotics-er-1.5-preview": (0.30, 2.50), "gemini-robotics-er-1.6-preview": (0.30, 2.50),
}


def key() -> str:
    return Path("~/.gemini_key").expanduser().read_text().strip().split("=", 1)[1]


def _part(x):
    if isinstance(x, Image.Image):
        b = io.BytesIO(); x.save(b, format="PNG")
        return {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(b.getvalue()).decode()}}
    return {"text": str(x)}


class Budget:
    """Runs Gemini calls while enforcing a hard USD cap; tallies exact spend."""
    def __init__(self, cap_usd: float, log_path: str | Path | None = None):
        self.cap = cap_usd; self.spent = 0.0; self.calls = 0
        self.k = key(); self.log_path = Path(log_path) if log_path else None; self.rows = []

    def cost(self, model: str, usage: dict) -> float:
        pin, pout = PRICE[model]
        out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return usage.get("promptTokenCount", 0) / 1e6 * pin + out / 1e6 * pout

    def call(self, model: str, contents: list, max_tokens: int = 400,
             thinking_budget: int | None = None, temperature: float = 0.0, tries: int = 4):
        if self.spent >= self.cap:
            raise RuntimeError(f"budget cap ${self.cap:.2f} reached (spent ${self.spent:.4f})")
        gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if thinking_budget is not None: gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        body = {"contents": [{"parts": [_part(c) for c in contents]}], "generationConfig": gen}
        for a in range(tries):
            try:
                r = requests.post(f"{API}/{model}:generateContent", params={"key": self.k}, json=body, timeout=120)
                if r.status_code == 200:
                    j = r.json(); usage = j.get("usageMetadata", {}); c = self.cost(model, usage)
                    self.spent += c; self.calls += 1; cand = j.get("candidates", [])
                    txt = "".join(p.get("text", "") for p in cand[0].get("content", {}).get("parts", [])) if cand else ""
                    self.rows.append({"model": model, "cost": c, "usage": usage})
                    if self.log_path:
                        self.log_path.write_text(json.dumps({"spent": self.spent, "calls": self.calls, "rows": self.rows}, indent=1))
                    return txt, usage
                if r.status_code == 429 and "spending cap" in r.text:
                    raise RuntimeError("gemini project spend cap hit (not budget guard)")
                if r.status_code in (429, 500, 503, 504): time.sleep(min(30, 3 * 2 ** a)); continue
                raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
            except requests.RequestException:
                if a == tries - 1: raise
                time.sleep(3)
        return "", {}


def parse_json(text: str):
    """First balanced-brace JSON object (handles nesting + ``` fences)."""
    if not text: return None
    t = text.strip(); start = t.find("{")
    if start < 0: return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{": depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(t[start:i + 1])
                except Exception: return None
    return None

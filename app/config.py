from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = Path(cfg.get("data", "data"))
    if not data.is_absolute():
        data = ROOT / data
    cfg["data"] = data
    (data / "audio").mkdir(parents=True, exist_ok=True)
    (data / "notes").mkdir(parents=True, exist_ok=True)
    return cfg

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def cargar(ruta: str | Path = RAIZ / "config.yaml") -> dict:
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datos = Path(cfg.get("datos", "datos"))
    if not datos.is_absolute():
        datos = RAIZ / datos
    cfg["datos"] = datos
    (datos / "audio").mkdir(parents=True, exist_ok=True)
    (datos / "apuntes").mkdir(parents=True, exist_ok=True)
    return cfg

"""Prueba sin web: pasa tests/clase_prueba.wav por el pipeline y muestra el resultado final.

  .venv/bin/python tests/prueba_pipeline.py [--tiempo-real]
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import cargar  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import Store  # noqa: E402

cfg = cargar()
cfg["datos"] = Path(tempfile.mkdtemp())
(cfg["datos"] / "audio").mkdir()
store = Store(cfg["datos"] / "prueba.db")
eventos = []
t0 = time.monotonic()


def emitir(ev):
    ev["_t"] = time.monotonic() - t0
    eventos.append(ev)
    if ev["tipo"] in ("estado", "frase", "hablante"):
        print(f"{ev['_t']:6.1f}s", {k: v for k, v in ev.items() if k != "_t"})


archivo = Path(__file__).parent / "clase_prueba.wav"
p = Pipeline(cfg, store, "Prueba", emitir, archivo=archivo, rapido="--tiempo-real" not in sys.argv)
p.start()
p.join()

print("\n=== Resultado final ===")
for f in store.frases(p.clase_id):
    print(f"[{f['t_inicio']:5.1f}] {f['hablante']} ({f['p_profe']:.2f})  {f['texto']}")
parciales = [e for e in eventos if e["tipo"] == "parcial" and e.get("texto")]
procs = [e["proc_ms"] for e in eventos if e["tipo"] == "frase"]
print(f"\nparciales: {len(parciales)} · proceso por frase: media {sum(procs)/len(procs):.0f} ms, máx {max(procs)} ms")
audio = list((cfg["datos"] / "audio").glob("*.opus"))
print("audio:", audio[0].name, f"{audio[0].stat().st_size/1024:.0f} KB" if audio else "NO")
print("huella guardada:", store.huella_profe(store.asignatura("Prueba")["id"]) is not None)

"""Phase 4 test without the web page: alert rules, a custom rule confirmed by Claude, and 👍/👎 learning.

  .venv/bin/python tests/test_alerts.py

Uses your Claude subscription only for the custom rule (a few hundred tokens per call).
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.alerts import Alerts  # noqa: E402
from app.config import load  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.llm import SubscriptionLLM  # noqa: E402
from app.store import Store  # noqa: E402

SENTENCES = [  # (t, expected alert rules, text)
    (0, "exam, emphasis", "Buenos días. Hoy vamos a empezar con las derivadas. Abrid el cuaderno y apuntad, porque esto es importante y entra en el examen."),
    (7, "definition", "La derivada de una función mide cómo cambia su valor cuando cambia la variable. Formalmente, es el límite del cociente incremental cuando h tiende a cero."),
    (40, "tasks", "Lo vemos con la definición de límite. Para el jueves, haced los ejercicios del tema 3, de 1 al 10."),
    (48, "formula", "Vamos a verlo. Con la definición, el cociente incremental de elevado a x es elevado a x por elevado a h menos uno partido por h. Ese límite vale uno cuando h tiende a cero, y por eso la derivada de elevado a x es elevado a x."),
    (80, "-", "Bueno, ahora vamos a pasar a otro ejemplo en la pizarra, que se vea bien desde atrás."),
    (110, "exam", "Este tipo de ejercicio es el que suele aparecer al final, así que practicadlo en casa."),
    (140, "custom", "Si tenemos seno de x al cuadrado, primero derivamos la función de fuera y luego multiplicamos por la derivada de la de dentro."),
    (230, "new_terms", "Hoy introducimos el jacobiano, la hessiana, los multiplicadores de Lagrange y la convexidad estricta."),
]

cfg = load()
cfg["notifications"] = {"enabled": False}
store = Store(Path(tempfile.mkdtemp()) / "t.db")
embedder = Embedder()


class P:
    pass


p = P()
p.subject = store.subject("Calc")
p.class_id = store.new_class(p.subject["id"], "es")
seen = []


async def show(ev):
    if ev["type"] == "alert":
        seen.append(ev)
        fb = {1: " 👍", -1: " 👎"}.get(ev["feedback"], "")
        print(f"   → ALERT #{ev['id']} [{ev['kind']}] {ev['rules']} {ev['reason']}{fb}")


async def run(alerts, items):
    for t, exp, text in items:
        sid = store.new_sentence(p.class_id, t, t + 5, text, "T", 0.9, 0)
        print(f"[{t:3}s] expected: {exp:15} {text[:70]}")
        await alerts.on_sentence({"id": sid, "t0": t, "text": text, "speaker": "T"}, p)


async def main():
    llm = SubscriptionLLM(model=cfg["llm"]["answer_model"], effort=cfg["llm"]["effort"])
    llm.warm_up()
    alerts = Alerts(cfg, store, embedder, llm, show)
    alerts.add_rule("Chain rule", "cuando explique la regla de la cadena o la derivada de una función compuesta", "important")
    await asyncio.sleep(6)
    await run(alerts, SENTENCES)

    print("\n--- 👎 on the 'tasks' alert, then a very similar sentence ---")
    tasks = next(a for a in seen if "tasks" in a["rules"])
    before = alerts.rules["tasks"]["threshold"]
    await alerts.feedback(tasks["id"], -1)
    print(f"tasks threshold {before} -> {alerts.rules['tasks']['threshold']}")
    n = len(seen)
    await run(alerts, [(300, "silenced", "Lo vemos con la definición del límite. Para el jueves, haced los ejercicios del tema 3, del 1 al 10.")])
    print("silenced ✔" if not any("tasks" in a["rules"] for a in seen[n:]) else "NOT silenced ✘")
    print("\nrules:", [(r["key"], r["threshold"], r["learned"]) for r in alerts.rules_view()])
    await llm.close()

asyncio.run(main())

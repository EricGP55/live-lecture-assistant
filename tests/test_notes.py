"""Phase 5 test without the web page: transcribes the sample class with alerts, a doubt and a
question, then writes the notes in both modes:
  - direct: short class, the whole transcript goes in one call
  - chunked: as in a long class, condensed every 30 s while it runs (forced here), then combined

  .venv/bin/python tests/test_notes.py

Uses your Claude subscription (a few thousand tokens).
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.alerts import Alerts  # noqa: E402
from app.config import load  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.llm import SubscriptionLLM  # noqa: E402
from app.notes import Notes  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import Store  # noqa: E402

cfg = load()
cfg["data"] = Path(tempfile.mkdtemp())
(cfg["data"] / "audio").mkdir()
cfg["notifications"] = {"enabled": False}
store = Store(cfg["data"] / "class.db")
embedder = Embedder()
p = Pipeline(cfg, store, "Cálculo", lambda ev: None, audio_file=Path(__file__).parent / "class_sample.wav",
             fast=True, embedder=embedder)
p.start()
p.join()
sentences = store.sentences(p.class_id)
print(f"Transcribed {len(sentences)} sentences.")


class FakeLive:  # what Notes.on_sentence needs from the pipeline
    def __init__(self, t):
        self.class_id, self.t = p.class_id, t

    def current_time(self):
        return self.t


async def main():
    llm = SubscriptionLLM(model=cfg["llm"]["answer_model"], effort=cfg["llm"]["effort"])
    llm.warm_up()

    async def quiet(ev):
        pass

    alerts = Alerts(cfg, store, embedder, llm, quiet)
    for s in sentences:
        await alerts.on_sentence({"id": s["id"], "t0": s["t_start"], "text": s["text"], "speaker": s["speaker"]}, p)
    # a doubt still open and a question asked during the class
    did = store.new_doubt(p.subject["id"], p.class_id, 20, "¿Qué es una integral?", embedder.query("¿Qué es una integral?"))
    store.update_doubt(did, missing="No se ha explicado qué es una integral.",
                       questions=json.dumps(["¿Qué es una integral y cuándo la veremos?"]))
    store.save_question(p.class_id, 45, "class", "¿Qué deberes hay?",
                        "Ejercicios 1–10 del tema 3, para el jueves [00:40].", [], 1300, 40)
    await asyncio.sleep(5)

    for mode in ("direct", "chunked"):
        c = dict(cfg)
        c["notes"] = {"chunk_minutes": 10, "direct_chars": 6000} if mode == "direct" \
            else {"chunk_minutes": 0.5, "direct_chars": 0}
        notes = Notes(c, store, llm)
        if mode == "chunked":
            store._exec("DELETE FROM summaries")
            for s in sentences:  # replay the class clock, as on_sentence would see it live
                await notes.on_sentence(FakeLive(s["t_end"]))
        before = dict(llm.totals)
        path = await notes.generate(p.class_id)
        spent = llm.totals["tokens_in"] + llm.totals["tokens_out"] - before["tokens_in"] - before["tokens_out"]
        print(f"\n==================== {mode.upper()} · {len(store.summaries(p.class_id))} chunk summaries · "
              f"final step {spent} tokens · {path}\n")
        print(Path(path).read_text(encoding="utf-8"))
    print("Claude calls in this test:", llm.totals)
    await llm.close()

asyncio.run(main())

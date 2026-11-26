"""Phase 3 test without the web page: replays the sample class sentence by sentence and writes
doubts at given moments, to check the doubt check, the live watcher and the suggestions.

  .venv/bin/python tests/test_doubts.py

Uses your Claude subscription (a few thousand tokens).
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load  # noqa: E402
from app.doubts import Doubts  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.llm import SubscriptionLLM  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.search import Retriever, fmt_t  # noqa: E402
from app.store import Store  # noqa: E402

# (class time in s, doubt). Expected: 1 answered later by the teacher (at ~30-40 s),
# 2 already answered when written, 3 never answered.
DOUBTS = [
    (12, "¿Por qué la derivada de e elevado a x es ella misma?"),
    (32, "¿Entra esto en el parcial?"),
    (32, "¿Qué es una integral?"),
]

cfg = load()
cfg["data"] = Path(tempfile.mkdtemp())
(cfg["data"] / "audio").mkdir()
cfg["notifications"] = {"enabled": False}
embedder = Embedder()

# 1) transcribe the whole sample into a scratch store
src = Store(cfg["data"] / "src.db")
p0 = Pipeline(cfg, src, "Test", lambda ev: None, audio_file=Path(__file__).parent / "class_sample.wav",
              fast=True, embedder=embedder)
p0.start()
p0.join()
sentences = src.sentences(p0.class_id)


# 2) replay it into a fresh store, as if live
class FakePipeline:
    def __init__(self, store):
        self.store = store
        self.subject = store.subject("Test")
        self.class_id = store.new_class(self.subject["id"], "es")
        self.t = 0.0

    def current_time(self):
        return self.t

    def is_alive(self):
        return True


live = Store(cfg["data"] / "live.db")
fp = FakePipeline(live)
live.set_profile(fp.subject["id"], "Me preocupa el examen y no entiendo bien las demostraciones.",
                 {"address": "informal", "length": "short"})
tokens = {"n": 0}


async def show(ev):
    if ev["type"] == "doubt" and not ev.get("checking"):
        tag = "HIT " if ev.get("hit") else "    "
        print(f"  {tag}[{ev['status']:9}] {ev['text']}\n        answer: {ev['answer'][:110]}"
              f"\n        evidence: {ev['evidence']}  missing: {ev['missing'][:60]}  questions: {ev['questions']}")
    elif ev["type"] == "moment":
        print(f"  MOMENT: {ev['text']}")


async def main():
    llm = SubscriptionLLM(model=cfg["llm"]["answer_model"], effort=cfg["llm"]["effort"])
    d = Doubts(cfg, live, Retriever(live, embedder), embedder, llm, show)
    llm.warm_up()
    await asyncio.sleep(6)
    pending = list(DOUBTS)
    for s in sentences:
        fp.t = s["t_start"]
        while pending and pending[0][0] <= fp.t:
            t, text = pending.pop(0)
            print(f"[{fmt_t(t)}] student writes doubt: {text}")
            await d.add(text, fp)
        sid = live.new_sentence(fp.class_id, s["t_start"], s["t_end"], s["text"], s["speaker"], s["p_teacher"], None)
        live.set_embedding(sid, embedder.passages([s["text"]])[0])
        ev = {"type": "sentence", "id": sid, "t0": s["t_start"], "text": s["text"], "speaker": s["speaker"]}
        print(f"[{fmt_t(s['t_start'])} {s['speaker']}] {s['text'][:90]}")
        await d.on_sentence(ev, fp)
        await asyncio.sleep(0.2)
        while d.confirming:
            await asyncio.sleep(0.5)
    print("\nsimilarity history per doubt:", {k: [round(x, 3) for x in v] for k, v in d.stats.items()})
    await asyncio.sleep(6)
    qs, err = await d.suggest(fp)
    print("\nSUGGESTED QUESTIONS:", *[f"\n  - {q['q']}  ({q['why']})" for q in qs], err or "")
    rows = live._query("SELECT text, status, tokens FROM doubts")
    print("\nfinal:", *[f"\n  {r['status']:9} {r['tokens']:5} tok  {r['text']}" for r in rows])
    await llm.close()

asyncio.run(main())

"""End-to-end test of phase 2 without the web page: transcribes the sample class, then asks
questions through the Claude subscription and prints answers, latency and tokens.

  .venv/bin/python tests/test_qa.py [haiku|sonnet]

Note: this uses your subscription (a few thousand tokens in total).
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.qa import QA  # noqa: E402
from app.search import Retriever  # noqa: E402
from app.store import Store  # noqa: E402

QUESTIONS = [
    "¿Qué deberes hay y para cuándo?",
    "¿Por qué la derivada de e^x es e^x?",
    "What did the teacher say about the exam?",
    "¿Qué dijo sobre las integrales?",
]

cfg = load()
cfg["data"] = Path(tempfile.mkdtemp())
(cfg["data"] / "audio").mkdir()
if len(sys.argv) > 1:
    cfg.setdefault("llm", {})["answer_model"] = sys.argv[1]
store = Store(cfg["data"] / "test.db")
embedder = Embedder()
p = Pipeline(cfg, store, "Test", lambda ev: None, audio_file=Path(__file__).parent / "class_sample.wav",
             fast=True, embedder=embedder)
p.start()
p.join()
print(f"Transcribed {len(store.sentences(p.class_id))} sentences. Model: {cfg['llm']['answer_model']}\n")


async def main():
    qa = QA(cfg, Retriever(store, embedder))
    qa.llm.warm_up()
    await asyncio.sleep(6)  # let the warm spare connect, as it would while the class runs
    for q in QUESTIONS:
        done = {}

        async def send(ev):
            if ev["type"] == "answer_delta":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "answer_done":
                done.update(ev)

        print(f"Q: {q}\nA: ", end="")
        await qa.ask(q, "class", p, send)
        print(f"\n   [first token {done.get('ttft_ms')} ms · in {done.get('tokens_in')} tok · "
              f"out {done.get('tokens_out')} tok · error {done.get('error')}]\n")
        await asyncio.sleep(6)  # time for the next spare, like a real pause between questions
    await qa.llm.close()

asyncio.run(main())

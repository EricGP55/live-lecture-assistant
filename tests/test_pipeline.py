"""Headless test: runs tests/class_sample.wav through the pipeline and prints the result.

  .venv/bin/python tests/test_pipeline.py [--realtime]
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import Store  # noqa: E402

cfg = load()
cfg["data"] = Path(tempfile.mkdtemp())
(cfg["data"] / "audio").mkdir()
store = Store(cfg["data"] / "test.db")
events = []
t0 = time.monotonic()


def emit(ev):
    ev["_t"] = time.monotonic() - t0
    events.append(ev)
    if ev["type"] in ("status", "sentence", "speaker"):
        print(f"{ev['_t']:6.1f}s", {k: v for k, v in ev.items() if k != "_t"})


audio = Path(__file__).parent / "class_sample.wav"
p = Pipeline(cfg, store, "Test", emit, audio_file=audio, fast="--realtime" not in sys.argv)
p.start()
p.join()

print("\n=== Final result ===")
for s in store.sentences(p.class_id):
    print(f"[{s['t_start']:5.1f}] {s['speaker']} ({s['p_teacher']:.2f})  {s['text']}")
partials = [e for e in events if e["type"] == "partial" and e.get("text")]
procs = [e["proc_ms"] for e in events if e["type"] == "sentence"]
print(f"\npartials: {len(partials)} · processing per sentence: mean {sum(procs)/len(procs):.0f} ms, max {max(procs)} ms")
files = list((cfg["data"] / "audio").glob("*.opus"))
print("audio:", f"{files[0].name} {files[0].stat().st_size/1024:.0f} KB" if files else "MISSING")
print("voiceprint saved:", store.teacher_voiceprint(store.subject("Test")["id"]) is not None)

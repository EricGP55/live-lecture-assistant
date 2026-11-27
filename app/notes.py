"""Notes after class (phase 5): a Markdown file per class with summary, likely exam content,
definitions, tasks and dates, unresolved doubts, your questions, alerts and stats.

Token plan:
- During the class, every `chunk_minutes` the new stretch of transcript is condensed into
  ~120 words of notes (one short call), so the end of the class only has to read summaries.
- Short classes (transcript under `direct_chars`) skip that and send the transcript directly.
- The final call returns JSON (summary, exam, definitions, tasks); doubts, questions, alerts
  and stats come straight from the database at 0 tokens.
"""
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from .llm import SubscriptionLLM, parse_json
from .search import fmt_t

CHUNK = """Task: condense this stretch of a class transcript into compact study notes.
- Same language as the class. Max 120 words. Bullet points.
- Keep definitions, formulas, examples, exam hints, homework and dates; cite the moment as [mm:ss].
- The transcript comes from speech recognition: silently fix obvious errors. T = teacher, S = student.

Transcript:
{lines}"""

FINAL = """Task: write the study notes of a class from the material below.
Reply with JSON only, in the language of the class:
{{"topic": "the main topic in a few words",
 "summary": ["5-8 bullets: what was explained, in order"],
 "exam": ["things the teacher said will or may be on the exam, or stressed as important ([] if none)"],
 "definitions": ["key definitions, rules and formulas, each one line ([] if none)"],
 "tasks": [{{"what": "homework / deliverable / logistics", "when": "date or deadline if said, else \\"\\""}}]}}
Cite moments as [mm:ss] when you know them. Do not invent anything that is not in the material.
The transcript comes from speech recognition: silently fix obvious errors.

Alerts raised during the class (useful hints):
{alerts}

Material ({kind}):
{material}"""

HEADINGS = {
    "es": {"summary": "Resumen", "exam": "Importante / puede entrar en el examen", "definitions": "Definiciones y fórmulas",
           "tasks": "Tareas y fechas", "open": "Dudas sin resolver", "answered": "Dudas resueltas en clase",
           "qa": "Tus preguntas durante la clase", "alerts": "Avisos", "stats": "Datos de la clase",
           "ask": "Para preguntar", "missing": "Falta", "none": "Nada.", "earlier": "de una clase anterior",
           "duration": "Duración", "sentences": "frases", "teacher": "habla el profesor", "tokens": "tokens de Claude",
           "audio": "Audio"},
    "en": {"summary": "Summary", "exam": "Important / may be on the exam", "definitions": "Definitions and formulas",
           "tasks": "Tasks and dates", "open": "Unresolved doubts", "answered": "Doubts answered in class",
           "qa": "Your questions during class", "alerts": "Alerts", "stats": "Class data",
           "ask": "To ask", "missing": "Missing", "none": "Nothing.", "earlier": "from an earlier class",
           "duration": "Duration", "sentences": "sentences", "teacher": "teacher speaking", "tokens": "Claude tokens",
           "audio": "Audio"},
}


def line(s) -> str:
    return f"[{fmt_t(s['t_start'])} {s['speaker'] or '?'}] {s['text']}"


class Notes:
    def __init__(self, cfg: dict, store, llm: SubscriptionLLM, broadcast=None):
        c = cfg.get("notes", {})
        self.cfg, self.store, self.llm, self.broadcast = cfg, store, llm, broadcast
        self.chunk_s = c.get("chunk_minutes", 10) * 60
        self.direct_chars = c.get("direct_chars", 6000)
        self.dir = Path(cfg["data"]) / "notes"
        self.lang = "es" if str(cfg.get("language", "es")).startswith("es") else "en"
        self._chunking = False
        self.done: dict[int, asyncio.Task] = {}

    # ---------- during the class
    async def on_sentence(self, pipeline):
        """Condense the last stretch once `chunk_minutes` of class have passed since the previous one."""
        if self._chunking:
            return
        done = await asyncio.to_thread(self.store.summaries, pipeline.class_id)
        start = done[-1]["t1"] if done else 0.0
        now = pipeline.current_time()
        if now - start < self.chunk_s:
            return
        self._chunking = True
        try:
            await self._summarize(pipeline.class_id, start, now)
        finally:
            self._chunking = False

    async def _summarize(self, class_id, t0, t1):
        lines = await asyncio.to_thread(self.store.sentences_between, class_id, t0, t1)
        if not lines:
            return
        ans = await self.llm.complete(CHUNK.format(lines="\n".join(line(s) for s in lines)))
        if ans.text and not ans.error:
            await asyncio.to_thread(self.store.add_summary, class_id, t0, t1, ans.text.strip(),
                                    ans.tokens_in + ans.tokens_out)

    # ---------- at the end
    def generate_once(self, class_id: int) -> asyncio.Task:
        """One generation per class, however many times it is requested (button + Ctrl+C)."""
        if class_id not in self.done:
            self.done[class_id] = asyncio.create_task(self.generate(class_id))
        return self.done[class_id]

    async def generate(self, class_id: int) -> str | None:
        st = self.store
        info = await asyncio.to_thread(st.class_info, class_id)
        sentences = await asyncio.to_thread(st.sentences, class_id)
        if not info or not sentences:
            return None
        if self.broadcast:
            await self.broadcast({"type": "notes", "status": "writing", "class_id": class_id})
        summaries = await asyncio.to_thread(st.summaries, class_id)
        full = "\n".join(line(s) for s in sentences)
        final_tokens = 0
        if not summaries and len(full) <= self.direct_chars:
            kind, material = "full transcript", full
        else:
            start = summaries[-1]["t1"] if summaries else 0.0
            if any(s["t_start"] >= start for s in sentences):
                await self._summarize(class_id, start, sentences[-1]["t_start"] + 1)
                summaries = await asyncio.to_thread(st.summaries, class_id)
            kind = "notes of each stretch of the class, in order"
            material = "\n\n".join(f"### {fmt_t(s['t0'])}–{fmt_t(s['t1'])}\n{s['text']}" for s in summaries)
        alerts = await asyncio.to_thread(st.alerts, class_id)
        alert_lines = "\n".join(f"[{fmt_t(a['t'])}] {a['reason']}: {(a.get('text') or '')[:160]}"
                                for a in alerts if a["feedback"] >= 0) or "(none)"
        ans = await self.llm.complete(FINAL.format(alerts=alert_lines, kind=kind, material=material))
        final_tokens = ans.tokens_in + ans.tokens_out
        data = parse_json(ans.text) or {}
        md = await asyncio.to_thread(self._render, info, sentences, data, alerts, final_tokens, ans.error)
        folder = self.dir / re.sub(r"[^\w\-]+", "_", info["subject"]).strip("_")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.fromtimestamp(info['started_at']):%Y-%m-%d_%H%M}.md"
        path.write_text(md, encoding="utf-8")
        await asyncio.to_thread(st.set_notes_path, class_id, str(path))
        if self.broadcast:
            await self.broadcast({"type": "notes", "status": "ready", "class_id": class_id,
                                  "url": f"/notes/{class_id}", "path": str(path), "error": ans.error})
        return str(path)

    def _render(self, info, sentences, data, alerts, final_tokens, error) -> str:
        h = HEADINGS[self.lang]
        start = datetime.fromtimestamp(info["started_at"])
        duration = sentences[-1]["t_end"] if sentences else 0
        topic = data.get("topic") or ""
        out = [f"# {info['subject']} — {start:%Y-%m-%d %H:%M}" + (f" · {topic}" if topic else ""), ""]
        if error and not data:
            out += [f"> ⚠ {error}", ""]

        def section(key, items, fmt=lambda x: x):
            out.append(f"## {h[key]}")
            out.extend(f"- {fmt(x)}" for x in items) if items else out.append(h["none"])
            out.append("")

        section("summary", data.get("summary", []))
        section("exam", data.get("exam", []))
        section("definitions", data.get("definitions", []))
        section("tasks", [t for t in data.get("tasks", []) if isinstance(t, dict) and t.get("what")],
                lambda t: f"**{t['when']}** — {t['what']}" if t.get("when") else t["what"])

        doubts = self.store.class_doubts(info["id"], info["subject_id"])
        open_d = [d for d in doubts if d["status"] in ("pending", "partial")]
        done_d = [d for d in doubts if d["status"] == "answered"]
        out.append(f"## {h['open']}")
        if not open_d:
            out.append(h["none"])
        for d in open_d:
            tag = f" _({h['earlier']})_" if d["class_id"] != info["id"] else ""
            out.append(f"- **{d['text']}**{tag}")
            if d["answer"]:
                out.append(f"  - {d['answer']}")
            if d["missing"]:
                out.append(f"  - {h['missing']}: {d['missing']}")
            for q in json.loads(d["questions"] or "[]"):
                out.append(f"  - {h['ask']}: _{q}_")
        out.append("")
        if done_d:
            out.append(f"## {h['answered']}")
            out.extend(f"- **{d['text']}** — {d['answer']}" for d in done_d)
            out.append("")

        qa = self.store.questions(info["id"])
        if qa:
            out.append(f"## {h['qa']}")
            for q in qa:
                out.append(f"- **{q['question']}**")
                out.append(f"  {(q['answer'] or '').strip()}".replace("\n", "\n  "))
            out.append("")

        useful = [a for a in alerts if a["feedback"] >= 0]
        if useful:
            out.append(f"## {h['alerts']}")
            out.extend(f"- `{fmt_t(a['t'])}` {'⭐' if a['kind'] == 'important' else '🧩'} {a['reason']} — "
                       f"{(a.get('text') or '')[:140]}" for a in useful)
            out.append("")

        tok = self.store.class_tokens(info["id"])
        total = sum(tok.values()) + final_tokens
        teacher = sum(1 for s in sentences if s["speaker"] == "T")
        out += [f"## {h['stats']}",
                f"- {h['duration']}: {fmt_t(duration)} · {len(sentences)} {h['sentences']} · "
                f"{round(100 * teacher / max(len(sentences), 1))}% {h['teacher']}",
                f"- {h['tokens']}: ~{total:,} (questions {tok['questions']:,} · doubts {tok['doubts']:,} · "
                f"summaries {tok['summaries'] + final_tokens:,})"]
        if info.get("audio_path"):
            out.append(f"- {h['audio']}: `{info['audio_path']}`")
        return "\n".join(out) + "\n"

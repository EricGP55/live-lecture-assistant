"""Doubts mode (phase 3).

- check:   "has the class already answered my doubt?" -> one Claude call returning JSON with
           status, what was said, what is missing and questions to ask.
- watch:   every new sentence is compared *locally* (embeddings, 0 tokens) with each open doubt;
           only a clear match is confirmed with a short Claude call, then the student is notified.
- suggest: 2-3 good questions for right now, from the student's concerns, open doubts and the last minutes.
- moments: "any questions?" from the teacher, or a long pause after the teacher speaks, while there
           are open doubts -> "good moment to ask" (local, 0 tokens).
"""
import asyncio
import json
import re
import time

import numpy as np

from .llm import SubscriptionLLM, parse_json
from .notify import notify
from .search import Retriever, fmt_t

OPEN = ("pending", "partial")

MOMENT_RE = re.compile(
    r"(¿?\s*(alguna|algunas|hay|tenéis|tienen|alguien tiene)\s+(otra\s+)?(pregunta|duda)s?"
    r"|¿\s*(preguntas|dudas)\s*\?|any questions|questions\?|está claro\??|queda claro|se entiende\??)",
    re.I)


def style_text(style: dict) -> str:
    address = ("address the teacher formally (usted)" if style.get("address") == "formal"
               else "address the teacher informally (tú)")
    length = ("you may connect to what the teacher just said, max 2 sentences each"
              if style.get("length") == "elaborate" else "one short, direct sentence each")
    return f"{address}; {length}"


CHECK = """Task: during the class the student wrote a doubt. Using ONLY the transcript excerpts, decide whether the class has already answered it.
Reply with JSON only:
{{"status": "answered" | "partial" | "not_answered",
 "answer": "what the class said about it, max 50 words, in the language of the doubt; say so if it was in an earlier class (\\"\\" if nothing)",
 "evidence": ["mm:ss" for the current class, "YYYY-MM-DD mm:ss" for an earlier class (see the ## headers)],
 "missing": "what is still unanswered, max 20 words (\\"\\" if answered)",
 "questions": ["1-2 questions the student could ask the teacher to resolve what is missing ([] if answered)"]}}
Questions: in the language of the doubt; {style}; anchor them to what the teacher said when possible; never ask what the excerpts already answer.
The transcript comes from speech recognition; silently fix obvious errors.

Transcript excerpts (now = {now}):
{context}

Doubt: {doubt}"""

CONFIRM = """Task: decide whether the teacher has just answered (fully or partly) the student's doubt.
Doubt: {doubt}{previous}
New transcript lines:
{lines}
Reply with JSON only: {{"answers": "yes" | "partial" | "no", "answer": "max 40 words in the language of the doubt (\\"\\" if no)"}}"""

SUGGEST = """Task: suggest 2-3 good questions the student could ask the teacher right now.
Student's concerns for this subject: {concerns}
Open doubts: {doubts}
Last minutes of the class (now = {now}):
{lines}
Rules: same language as the class; {style}; prioritise the open doubts and concerns; anchor to what was just said; skip anything the transcript already answers; most useful first.
Reply with JSON only: {{"questions": [{{"q": "...", "why": "max 12 words"}}]}}"""


class Doubts:
    def __init__(self, cfg: dict, store, retriever: Retriever, embedder, llm: SubscriptionLLM, broadcast):
        c = cfg.get("doubts", {})
        self.store, self.retriever, self.embedder, self.llm = store, retriever, embedder, llm
        self.broadcast = broadcast                     # async fn(event) -> all clients
        self.floor = c.get("match_floor", 0.85)        # never confirm below this similarity
        self.strong = c.get("match_strong", 0.88)      # clearly about the doubt: confirm right away
        self.z = c.get("match_z", 2.0)                 # or this many std devs above the doubt's usual
        self.cooldown = c.get("confirm_cooldown_s", 30)
        self.pause_s = c.get("pause_moment_s", 4)
        self.notify_on = cfg.get("notifications", {}).get("enabled", True)
        self.window: list[dict] = []                   # last sentences (for matching)
        self.stats: dict[int, list[float]] = {}        # similarity history per doubt
        self.last_confirm: dict[int, float] = {}
        self.confirming: set[int] = set()
        self.last_moment = 0.0

    # ---------- helpers
    def _view(self, d: dict, pipeline) -> dict:
        return {"id": d["id"], "text": d["text"], "status": d["status"], "answer": d["answer"] or "",
                "evidence": json.loads(d["evidence"] or "[]"), "missing": d["missing"] or "",
                "questions": json.loads(d["questions"] or "[]"),
                "earlier": d["class_id"] != pipeline.class_id, "answered_t": d["answered_t"]}

    async def _emit(self, doubt_id: int, pipeline, **extra):
        d = await asyncio.to_thread(self.store.doubt, doubt_id)
        if d:
            await self.broadcast({"type": "doubt", **self._view(d, pipeline), **extra})

    def _notify(self, title, text):
        if self.notify_on:
            notify(title, text)

    def visible(self, pipeline) -> list[dict]:
        """This class's doubts (any status) plus unresolved ones from earlier classes."""
        return self.store.open_doubts(pipeline.subject["id"], pipeline.class_id)

    def open_list(self, pipeline) -> list[dict]:
        """Only doubts still waiting for an answer."""
        return [d for d in self.visible(pipeline) if d["status"] in OPEN]

    def views(self, pipeline) -> list[dict]:
        return [self._view(d, pipeline) for d in self.visible(pipeline)]

    # ---------- check
    async def add(self, text: str, pipeline):
        emb = await asyncio.to_thread(self.embedder.query, text)
        did = await asyncio.to_thread(self.store.new_doubt, pipeline.subject["id"], pipeline.class_id,
                                      pipeline.current_time(), text, emb)
        await self.check(did, pipeline)

    async def check(self, doubt_id: int, pipeline):
        d = await asyncio.to_thread(self.store.doubt, doubt_id)
        await self._emit(doubt_id, pipeline, checking=True)
        context, _ = await asyncio.to_thread(
            self.retriever.retrieve, d["text"], "subject", pipeline.class_id, pipeline.subject["id"],
            max_chars=4000, live=pipeline.is_alive())
        if not context:
            await self._emit(doubt_id, pipeline)
            return
        prof = await asyncio.to_thread(self.store.profile, pipeline.subject["id"])
        ans = await self.llm.complete(CHECK.format(style=style_text(prof["style"]), context=context,
                                                   now=fmt_t(pipeline.current_time()), doubt=d["text"]))
        data = parse_json(ans.text) or {}
        status = {"answered": "answered", "partial": "partial"}.get(data.get("status"), "pending")
        fields = dict(status=status, answer=str(data.get("answer", ""))[:600],
                      evidence=json.dumps(data.get("evidence", [])[:6]),
                      missing=str(data.get("missing", ""))[:300],
                      questions=json.dumps([str(q) for q in data.get("questions", [])][:3]),
                      tokens=(d["tokens"] or 0) + ans.tokens_in + ans.tokens_out)
        if status == "answered":
            fields.update(answered_class_id=pipeline.class_id, answered_t=pipeline.current_time())
        await asyncio.to_thread(self.store.update_doubt, doubt_id, **fields)
        await self._emit(doubt_id, pipeline, error=ans.error or (None if data else "Could not read the answer"))

    async def dismiss(self, doubt_id: int, pipeline):
        await asyncio.to_thread(self.store.update_doubt, doubt_id, status="dismissed")
        await self._emit(doubt_id, pipeline)

    # ---------- watch (called for every new sentence)
    async def on_sentence(self, ev: dict, pipeline):
        self.window = (self.window + [ev])[-3:]
        await self._maybe_moment(ev, pipeline)
        open_doubts = await asyncio.to_thread(self.open_list, pipeline)
        if not open_doubts or ev.get("speaker") == "S":
            return
        # Compare only the newest sentence: measured on the sample class, sentences that really
        # address a doubt score 0.90-0.91 while the rest stay at <= 0.85 (a 3-sentence window blurs this).
        vec = (await asyncio.to_thread(self.embedder.passages, [ev["text"]]))[0]
        now = time.monotonic()
        for d in open_doubts:
            sim = float(np.frombuffer(d["embedding"], dtype=np.float32) @ vec)
            hist = self.stats.setdefault(d["id"], [])
            mean, std = (np.mean(hist), np.std(hist)) if len(hist) >= 4 else (None, None)
            hist.append(sim)
            strong = sim >= self.strong
            unusual = mean is not None and sim >= mean + self.z * max(std, 0.005)
            if sim < self.floor or not (strong or unusual) or d["id"] in self.confirming:
                continue
            # cooldown between confirmations of the same doubt, skipped for a strong match
            if now - self.last_confirm.get(d["id"], 0.0) < self.cooldown and not strong:
                continue
            self.last_confirm[d["id"]] = now
            self.confirming.add(d["id"])
            asyncio.create_task(self._confirm(d, list(self.window), pipeline))

    async def _confirm(self, d: dict, window: list[dict], pipeline):
        try:
            prev = f"\nAlready known (partial answer): {d['answer']}" if d["status"] == "partial" and d["answer"] else ""
            lines = "\n".join(f"[{fmt_t(s['t0'])} {s.get('speaker') or '?'}] {s['text']}" for s in window)
            ans = await self.llm.complete(CONFIRM.format(doubt=d["text"], previous=prev, lines=lines))
            data = parse_json(ans.text) or {}
            verdict = data.get("answers")
            tokens = (d["tokens"] or 0) + ans.tokens_in + ans.tokens_out
            if verdict not in ("yes", "partial"):
                await asyncio.to_thread(self.store.update_doubt, d["id"], tokens=tokens)
                return
            t = window[-1]["t0"]
            evidence = json.loads(d["evidence"] or "[]") + [fmt_t(t)]  # the sentence that matched
            answer = data.get("answer", "")
            if d["status"] == "partial" and d["answer"] and verdict == "partial":
                answer = f"{d['answer']} {answer}".strip()
            fields = dict(status="answered" if verdict == "yes" else "partial", answer=answer[:600],
                          evidence=json.dumps(list(dict.fromkeys(evidence))[-6:]), tokens=tokens)
            if verdict == "yes":
                fields.update(missing="", questions="[]", answered_class_id=pipeline.class_id, answered_t=t)
            await asyncio.to_thread(self.store.update_doubt, d["id"], **fields)
            await self._emit(d["id"], pipeline, hit=True)
            what = "is being answered" if verdict == "yes" else "is being partly answered"
            self._notify("Your doubt " + what, f"{d['text'][:80]} ({fmt_t(t)})")
        finally:
            self.confirming.discard(d["id"])

    # ---------- good moment to ask (local)
    async def _maybe_moment(self, ev, pipeline, reason=None):
        if reason is None:
            if ev.get("speaker") == "S" or not MOMENT_RE.search(ev.get("text", "")):
                return
            reason = "The teacher is asking for questions"
        if time.monotonic() - self.last_moment < 90:
            return
        n_open = len(await asyncio.to_thread(self.open_list, pipeline))
        if not n_open:
            return
        self.last_moment = time.monotonic()
        text = f"{reason}. You have {n_open} open doubt{'s' if n_open > 1 else ''}."
        await self.broadcast({"type": "moment", "text": text})
        self._notify("Good moment to ask", text)

    async def on_pause(self, ev, pipeline):
        if self.window and self.window[-1].get("speaker") == "T":
            await self._maybe_moment(ev, pipeline, reason=f"The teacher paused for {int(ev['after_s'])} s")

    # ---------- suggest
    async def suggest(self, pipeline) -> tuple[list[dict], str | None]:
        now = pipeline.current_time()
        lines = await asyncio.to_thread(self.store.sentences_between, pipeline.class_id, max(0, now - 180), now)
        if not lines:
            return [], "Nothing has been said yet."
        prof = await asyncio.to_thread(self.store.profile, pipeline.subject["id"])
        doubts = [d["text"] for d in await asyncio.to_thread(self.open_list, pipeline)]
        ans = await self.llm.complete(SUGGEST.format(
            concerns=prof["concerns"] or "(none given)", doubts="; ".join(doubts) or "(none)",
            now=fmt_t(now), style=style_text(prof["style"]),
            lines="\n".join(f"[{fmt_t(s['t_start'])} {s['speaker'] or '?'}] {s['text']}" for s in lines)))
        data = parse_json(ans.text) or {}
        qs = [{"q": str(x.get("q", "")), "why": str(x.get("why", ""))}
              for x in data.get("questions", []) if isinstance(x, dict) and x.get("q")][:3]
        return qs, ans.error or (None if qs else "Could not read the suggestions")

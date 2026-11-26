"""Questions about the class: retrieve the relevant transcript, ask Claude, stream the answer."""
import asyncio

from .llm import SubscriptionLLM
from .search import Retriever, fmt_t

SYSTEM_PROMPT = """You help a student during a live class. You get excerpts of the automatic transcript and a question.
Transcript lines look like "[mm:ss T] text": T = teacher, S = student, ? = unknown speaker.
Rules:
- Answer in the language of the question. Be brief: at most 80 words unless the student asks for more.
- Base the answer on the excerpts and cite the moments you use as [mm:ss] (for an earlier class: [YYYY-MM-DD mm:ss]).
- The transcript comes from speech recognition and may contain errors; silently fix obvious ones (e.g. "elevado a xs" means e^x).
- If the excerpts do not answer the question, say so in one short sentence. You may then add a brief general explanation, starting with "Fuera de la clase:" (or "Outside the class:" in English).
- No preamble, no restating the question."""


class QA:
    def __init__(self, cfg: dict, retriever: Retriever):
        llm_cfg = cfg.get("llm", {})
        self.retriever = retriever
        self.llm = SubscriptionLLM(SYSTEM_PROMPT, model=llm_cfg.get("answer_model", "sonnet"),
                                   effort=llm_cfg.get("effort", "low"))
        self.max_chars = llm_cfg.get("context_chars", 5000)

    async def ask(self, question: str, scope: str, pipeline, send):
        """send(event) delivers events to the asking client only."""
        store = pipeline.store
        context, ids = await asyncio.to_thread(
            self.retriever.retrieve, question, scope, pipeline.class_id, pipeline.subject["id"],
            max_chars=self.max_chars, live=pipeline.is_alive())
        now = pipeline.current_time()
        await send({"type": "answer_start", "scope": scope, "sources": len(ids)})
        if not context:
            text = "Nothing has been transcribed in this scope yet."
            await send({"type": "answer_delta", "text": text})
            await send({"type": "answer_done", "error": None, "tokens_in": 0, "tokens_out": 0})
            return
        prompt = f"Transcript excerpts (now = {fmt_t(now)}):\n{context}\n\nQuestion: {question}"

        async def on_delta(t):
            await send({"type": "answer_delta", "text": t})

        ans = await self.llm.stream(prompt, on_delta)
        await asyncio.to_thread(store.save_question, pipeline.class_id, now, scope, question,
                                ans.text, ids, ans.tokens_in, ans.tokens_out)
        await send({"type": "answer_done", "error": ans.error, "ttft_ms": ans.ttft_ms,
                    "tokens_in": ans.tokens_in, "tokens_out": ans.tokens_out})

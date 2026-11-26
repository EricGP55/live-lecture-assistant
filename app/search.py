"""Hybrid retrieval over the transcript: dense (e5 embeddings) + keyword (FTS5/BM25),
fused with Reciprocal Rank Fusion, expanded with neighbouring sentences and packed
into a compact, token-cheap context block.
"""
import re
from datetime import datetime

import numpy as np

from .embeddings import Embedder
from .store import Store

RRF_K = 60
STOP = set("""a al algo como con cual cuando de del el ella ellos en entre era es esa ese eso esta este esto
fue ha hay la las le lo los mas me mi muy no nos o para pero por que se si sin sobre su sus te tu un una
uno y ya the a an and are as at be but by for from how in is it of on or that the this to was what when
where which who why with""".split())


def fmt_t(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def fts_query(question: str) -> str:
    words = [w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3 and w not in STOP]
    return " OR ".join(f'"{w}"' for w in words[:12])


class Retriever:
    def __init__(self, store: Store, embedder: Embedder):
        self.store, self.emb = store, embedder

    def _ensure_embeddings(self, rows):
        missing = [r for r in rows if r["embedding"] is None]
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            vecs = self.emb.passages([r["text"] for r in batch])
            for r, v in zip(batch, vecs):
                self.store.set_embedding(r["id"], v)
                r["embedding"] = v.tobytes()

    def retrieve(self, question: str, scope: str, class_id: int, subject_id: int,
                 top_k=6, window=1, recent=6, max_chars=5000, live=True) -> tuple[str, list[int]]:
        """Returns (context text, ids of the sentences included)."""
        rows = self.store.scope_rows(scope, class_id, subject_id)
        if not rows:
            return "", []
        self._ensure_embeddings(rows)
        by_id = {r["id"]: i for i, r in enumerate(rows)}

        # dense ranking
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        sims = mat @ self.emb.query(question)
        dense = [rows[i]["id"] for i in np.argsort(-sims)[:20]]
        # keyword ranking
        q = fts_query(question)
        sparse = self.store.fts(q, scope, class_id, subject_id) if q else []

        score: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, sid in enumerate(ranking):
                score[sid] = score.get(sid, 0) + 1 / (RRF_K + rank)
        hits = sorted(score, key=score.get, reverse=True)[:top_k]

        # expand with neighbours in the same class
        chosen: set[int] = set()
        for sid in hits:
            i = by_id[sid]
            for j in range(i - window, i + window + 1):
                if 0 <= j < len(rows) and rows[j]["class_id"] == rows[i]["class_id"]:
                    chosen.add(j)
        # during a live class, always include the last few sentences ("what did they just say?")
        if live and scope != "all":
            current = [i for i, r in enumerate(rows) if r["class_id"] == class_id]
            chosen.update(current[-recent:])

        # pack chronologically, grouped by class, within the character budget (rank order decides what stays)
        priority = {by_id[s]: k for k, s in enumerate(hits)}
        ordered = sorted(chosen, key=lambda j: priority.get(j, 999))
        kept, used = [], 0
        for j in ordered:
            line_len = len(rows[j]["text"]) + 12
            if used + line_len > max_chars:
                continue
            kept.append(j)
            used += line_len
        kept.sort()

        out, last_class = [], None
        for j in kept:
            r = rows[j]
            if r["class_id"] != last_class:
                last_class = r["class_id"]
                day = datetime.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d")
                tag = "current class" if r["class_id"] == class_id else "earlier class"
                if out:
                    out.append("…")
                out.append(f"## {day} ({tag})")
            elif out and j - 1 not in kept:
                out.append("…")
            out.append(f"[{fmt_t(r['t_start'])} {r['speaker'] or '?'}] {r['text']}")
        return "\n".join(out), [rows[j]["id"] for j in kept]


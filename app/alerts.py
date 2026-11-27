"""Alerts (phase 4): tell the student when something important or complex is happening.

Layer 0 — local, 0 tokens, every teacher sentence:
  - text rules (regex, Spanish + English) for exam, emphasis, tasks and definitions -> alert directly
  - semantic match against example phrases (and the student's 👍 examples) -> candidate
  - repeated idea: the teacher says something very similar for the 3rd time -> alert
  - complexity: many new technical words in a short time, formulas / multi-step reasoning,
    or a topic close to the student's concerns profile -> alert
Layer 1 — Claude, only for semantic candidates without a clear text rule: one short call that
  answers which (if any) of the candidate rules really apply.
Learning — 👍/👎 on an alert stores the sentence as a positive/negative example of its rules and
  nudges each rule's threshold. A 👎'd sentence also silences very similar text-rule hits later.
"""
import asyncio
import json
import re
import time

import numpy as np

from .llm import SubscriptionLLM, parse_json
from .notify import notify
from .search import fmt_t

I = re.IGNORECASE

BUILTIN = [
    {"key": "exam", "kind": "important", "name": "Exam & grading", "threshold": 0.88,
     "description": "what will be on the exam, how it is graded, exam or test dates",
     "regex": [r"(entra|entrar[aá]|cae|caer[aá]|sale|saldr[aá]|pregunt\w*|vendr[aá]).{0,50}\b(examen|parcial|final|prueba|test)\b",
               r"\b(examen|parcial|final)\b.{0,50}\b(entra|cae|caer[aá]|pregunt\w*|vale|cuenta|nota)\b",
               r"\b(lo|la|los|las) voy a preguntar\b", r"\b(on|in) the (exam|test|final|midterm)\b",
               r"\b(suele|suelen|puede|pueden|podría|podrían)\s+(salir|aparecer|caer|entrar)\b",
               r"\b(tends to|usually|could|might) (come up|appear)\b",
               r"\bcuenta (para|en) (la )?nota\b", r"\bwill be (graded|assessed)\b"],
     "seeds": ["esto entra en el examen", "lo voy a preguntar en el parcial", "el examen será el día quince",
               "esto cuenta para la nota final", "this will be on the exam", "the midterm is next week"]},
    {"key": "emphasis", "kind": "important", "name": "Teacher's emphasis", "threshold": 0.89,
     "description": "the teacher stresses that something is important or must be remembered",
     "regex": [r"\b(muy |super |súper )?importante\b", r"\bfijaos( bien)?\b", r"\bapuntad\b", r"\bapuntadlo\b",
               r"\bno os olvidéis\b", r"\bquedaos con\b", r"\bojo con\b", r"\bes clave\b", r"\bimprescindible\b",
               r"\b(really |very )?important\b", r"\bwrite (this|it) down\b", r"\bpay attention\b", r"\bkey point\b"],
     "seeds": ["esto es muy importante, apuntadlo", "fijaos bien en esto", "no os olvidéis de esto",
               "quedaos con esta idea", "this is really important, write it down", "pay attention to this"]},
    {"key": "tasks", "kind": "important", "name": "Homework & logistics", "threshold": 0.89,
     "description": "homework, assignments, deadlines, changes of room or schedule, material to download",
     "regex": [r"\b(haced|hacer|entregad|entregar|traed|traer|subid|subir|leed|leer|estudiad|repasad)\b.{0,80}"
               r"\b(lunes|martes|miércoles|jueves|viernes|mañana|próximo día|próxima semana|semana que viene)\b",
               r"\bpara el (lunes|martes|miércoles|jueves|viernes|próximo día)\b",
               r"\b(deberes|entrega|fecha límite|plazo)\b", r"\bcambi\w* de (aula|clase|horario)\b",
               r"\b(homework|assignment|due (on|by|next))\b"],
     "seeds": ["para el jueves haced los ejercicios del tema tres", "la práctica se entrega el viernes",
               "la clase del lunes será en otra aula", "homework for next week is exercise five"]},
    {"key": "definition", "kind": "important", "name": "Key definitions", "threshold": 0.90,
     "description": "a formal definition, theorem, rule or key property is stated",
     "regex": [r"\bse define\b", r"\bformalmente\b", r"\bpor definición\b", r"\bteorema\b", r"\bse llama\b",
               r"\bes la única\b", r"\bis defined as\b", r"\bby definition\b", r"\btheorem\b"],
     "seeds": ["formalmente, la derivada se define como el límite", "el teorema dice que",
               "se llama función continua a la que", "by definition, a function is continuous if"]},
    {"key": "repeated", "kind": "important", "name": "Repeated idea", "threshold": 0.93,
     "description": "the teacher has repeated the same idea several times"},
    {"key": "new_terms", "kind": "complex", "name": "Many new terms", "threshold": 5,
     "description": "many new technical words in a short time"},
    {"key": "formula", "kind": "complex", "name": "Formulas & steps", "threshold": 6,
     "description": "a formula, derivation or multi-step procedure is being explained"},
    {"key": "concerns", "kind": "complex", "name": "Touches your concerns", "threshold": 0.875,
     "description": "the topic is close to what worries you in this subject (Profile tab)"},
]
BY_KEY = {r["key"]: r for r in BUILTIN}
STEP = {"repeated": 0.005, "new_terms": 1, "formula": 1}  # 👎 raises by this, 👍 lowers by half
BOUNDS = {"repeated": (0.88, 0.98), "new_terms": (3, 12), "formula": (4, 15)}
SIM_BOUNDS = (0.84, 0.95)

# Operator-like words and single-letter variables. Deliberately excludes topic nouns ("derivada",
# "función") and everyday words ("por", "más", "a", "y"), which made every sentence look like a formula.
MATH = re.compile(
    r"\b(igual a|elevad[oa]|partido por|dividido|multiplicado|raíz|al cuadrado|al cubo|tiende a|"
    r"logaritmo|despejamos|despejando|sustituimos|sustituyendo|cociente|sumatorio|"
    r"equals|squared|cubed|divided by|times|tends to|square root|log of|we substitute|[xyzhn]\s*(sub|prima)|[xhn])\b", I)
COMMON = set("""porque entonces también siempre ejemplo hacemos tenemos tenéis vamos estamos podemos cuando donde
mientras después antes ahora siguiente anterior primero segundo tercero cualquier ninguna alguien nosotros
vosotros ellos ustedes pregunta preguntas respuesta importante cuaderno ejercicios ejercicio examen parcial
because example something anything everything another important question questions""".split())
WORD = re.compile(r"[a-záéíóúüñ]{8,}", I)

CONFIRM = """Task: a live-class assistant warns the student about moments they must not miss. Decide which of these alert rules really apply to the LAST line (the earlier lines are context).
Rules:
{rules}
{examples}Transcript:
{lines}
Reply with JSON only: {{"alert": [rule keys that apply, or empty], "why": "max 10 words, in the language of the class"}}"""


class Alerts:
    def __init__(self, cfg: dict, store, embedder, llm: SubscriptionLLM, broadcast):
        c = cfg.get("alerts", {})
        self.store, self.embedder, self.llm, self.broadcast = store, embedder, llm, broadcast
        self.enabled = c.get("enabled", True)
        self.cooldown = c.get("rule_cooldown_s", 20)
        self.negative_sim = c.get("negative_match", 0.93)
        self.warmup = c.get("new_terms_warmup_s", 90)
        self.min_words = c.get("semantic_min_words", 6)
        self.notify_on = cfg.get("notifications", {}).get("enabled", True)
        self._seed_vecs: dict[str, np.ndarray] = {}
        self._concern = ("", None)
        self.history: list[tuple[float, np.ndarray]] = []   # (t, emb) of teacher sentences this class
        self.repeated_ideas: list[np.ndarray] = []
        self.vocab: set[str] = set()
        self.recent_new: list[tuple[float, int]] = []
        self.last_fired: dict[str, float] = {}
        self.window: list[dict] = []
        self._init_rules()

    # ---------- rules and examples
    def _init_rules(self):
        for r in BUILTIN:
            self.store.upsert_rule(r["key"], r["kind"], r["name"], r["description"], r["threshold"], True)
            if r.get("seeds"):
                self._seed_vecs[r["key"]] = self.embedder.passages(r["seeds"])
        self.reload()

    def reload(self):
        self.rules = {r["key"]: r for r in self.store.alert_rules()}
        ex = self.store.examples()
        self.pos = {k: np.stack([np.frombuffer(e["embedding"], dtype=np.float32) for e in ex
                                 if e["rule_key"] == k and e["polarity"] > 0]) for k in {e["rule_key"] for e in ex if e["polarity"] > 0}}
        self.neg = {k: np.stack([np.frombuffer(e["embedding"], dtype=np.float32) for e in ex
                                 if e["rule_key"] == k and e["polarity"] < 0]) for k in {e["rule_key"] for e in ex if e["polarity"] < 0}}
        self.example_text = {}
        for e in ex[-60:]:
            self.example_text.setdefault(e["rule_key"], []).append(("+" if e["polarity"] > 0 else "−", e["text"][:120]))

    def rules_view(self) -> list[dict]:
        out = []
        for r in self.rules.values():
            n_pos = len(self.pos.get(r["key"], [])); n_neg = len(self.neg.get(r["key"], []))
            out.append({"key": r["key"], "kind": r["kind"], "name": r["name"], "description": r["description"],
                        "enabled": bool(r["enabled"]), "builtin": bool(r["builtin"]),
                        "threshold": round(r["threshold"], 3), "learned": f"{n_pos}👍 {n_neg}👎"})
        return out

    def add_rule(self, name: str, description: str, kind: str):
        key = f"custom_{int(time.time() * 1000)}"
        emb = self.embedder.query(description)
        self.store.upsert_rule(key, "complex" if kind == "complex" else "important", name[:60],
                               description[:300], 0.86, False, emb)
        self.reload()

    def _negative_hit(self, key, vec) -> bool:
        neg = self.neg.get(key)
        return neg is not None and float((neg @ vec).max()) >= self.negative_sim

    def _semantic(self, key, vec) -> float:
        r = self.rules[key]
        scores = []
        if key in self._seed_vecs:
            scores.append(float((self._seed_vecs[key] @ vec).max()))
        if r["embedding"]:
            scores.append(float(np.frombuffer(r["embedding"], dtype=np.float32) @ vec))
        if key in self.pos:
            scores.append(float((self.pos[key] @ vec).max()))
        return max(scores) if scores else 0.0

    # ---------- per sentence
    async def on_sentence(self, ev: dict, pipeline):
        self.window = (self.window + [ev])[-3:]
        if not self.enabled or ev.get("speaker") == "S" or len(ev["text"]) < 12:
            return
        vec = (await asyncio.to_thread(self.embedder.passages, [ev["text"]]))[0]
        fired, scores, candidates = self.evaluate(ev, vec, pipeline)
        if candidates:
            fired_llm, why = await self._confirm(candidates, list(self.window))
            fired.update({k: why for k in fired_llm})
        now = ev["t0"]
        fired = {k: v for k, v in fired.items() if now - self.last_fired.get(k, -1e9) >= self.cooldown}
        if not fired:
            return
        for k in fired:
            self.last_fired[k] = now
        await self._raise(ev, fired, pipeline, "semantic+claude" if candidates and any(k in candidates for k in fired) else "rule")

    def evaluate(self, ev, vec, pipeline):
        """Local layer. Returns ({rule_key: why}, debug scores, [semantic candidates for Claude])."""
        text, t = ev["text"], ev["t0"]
        fired, scores, candidates = {}, {}, []
        active = {k: r for k, r in self.rules.items() if r["enabled"]}
        # text rules + semantic candidates
        for key, r in active.items():
            b = BY_KEY.get(key)
            if b and "regex" in b and any(re.search(p, text, I) for p in b["regex"]):
                scores[key] = "regex"
                if not self._negative_hit(key, vec):
                    fired[key] = b["name"]
                continue
            if ((b and "seeds" in b) or not r["builtin"]) and len(text.split()) >= self.min_words:
                s = self._semantic(key, vec)
                scores[key] = round(s, 3)
                if s >= r["threshold"] and not self._negative_hit(key, vec):
                    candidates.append(key)
        # repeated idea
        if "repeated" in active:
            thr = active["repeated"]["threshold"]
            same = [pt for pt, pv in self.history if t - pt > 30 and float(pv @ vec) >= thr]
            scores["repeated"] = len(same)
            already = any(float(iv @ vec) >= thr for iv in self.repeated_ideas)
            if len(same) >= 2 and not already:
                self.repeated_ideas.append(vec)
                fired["repeated"] = f"Repeated {len(same) + 1} times"
        self.history.append((t, vec))
        # new technical words
        words = {w.lower() for w in WORD.findall(text)} - COMMON
        new = words - self.vocab
        self.vocab |= words
        self.recent_new = [(pt, n) for pt, n in self.recent_new if t - pt <= 30] + [(t, len(new))]
        burst = sum(n for _, n in self.recent_new)
        scores["new_terms"] = burst
        if "new_terms" in active and t >= self.warmup and burst >= active["new_terms"]["threshold"]:
            fired["new_terms"] = f"{burst} new terms in 30 s"
            self.recent_new = []
        # formulas / steps
        n_math = len(MATH.findall(text))
        scores["formula"] = n_math
        if "formula" in active and n_math >= active["formula"]["threshold"]:
            fired["formula"] = "Formula or multi-step reasoning"
        # concerns profile
        if "concerns" in active:
            concerns = self.store.profile(pipeline.subject["id"])["concerns"].strip()
            if concerns:
                if self._concern[0] != concerns:
                    self._concern = (concerns, self.embedder.query(concerns))
                s = float(self._concern[1] @ vec)
                scores["concerns"] = round(s, 3)
                if s >= active["concerns"]["threshold"] and not self._negative_hit("concerns", vec):
                    fired["concerns"] = "Close to your concerns"
        return fired, scores, [c for c in candidates if c not in fired]

    async def _confirm(self, keys, window):
        rules = "\n".join(f"- {k}: {self.rules[k]['description']}" for k in keys)
        ex = []
        for k in keys:
            for sign, txt in self.example_text.get(k, [])[-3:]:
                ex.append(f"  {k} {sign} \"{txt}\"")
        examples = ("Student feedback on earlier alerts (+ useful, − not useful):\n" + "\n".join(ex) + "\n") if ex else ""
        lines = "\n".join(f"[{fmt_t(s['t0'])} {s.get('speaker') or '?'}] {s['text']}" for s in window)
        ans = await self.llm.complete(CONFIRM.format(rules=rules, examples=examples, lines=lines))
        data = parse_json(ans.text) or {}
        return [k for k in data.get("alert", []) if k in keys], str(data.get("why", ""))[:120]

    async def _raise(self, ev, fired: dict, pipeline, source):
        kinds = {self.rules[k]["kind"] for k in fired}
        kind = "important" if "important" in kinds else "complex"
        names = [self.rules[k]["name"] for k in fired]
        whys = [w for w in fired.values() if w and w not in names]
        reason = " · ".join(names) + (f" — {whys[0]}" if whys else "")
        aid = await asyncio.to_thread(self.store.new_alert, pipeline.class_id, ev["id"], ev["t0"], kind,
                                      list(fired), reason, source)
        await self.broadcast(self.view(await asyncio.to_thread(self.store.alert, aid)))
        if self.notify_on:
            notify(("⭐ Important" if kind == "important" else "🧩 Complex") + f" ({fmt_t(ev['t0'])})",
                   f"{reason}: {ev['text'][:120]}")

    def view(self, a: dict) -> dict:
        keys = json.loads(a["rules"])
        return {"type": "alert", "id": a["id"], "sentence_id": a["sentence_id"], "t": a["t"], "kind": a["kind"],
                "rules": keys, "reason": a["reason"], "feedback": a["feedback"], "text": a.get("text") or ""}

    # ---------- learning
    async def feedback(self, alert_id: int, value: int):
        a = await asyncio.to_thread(self.store.alert, alert_id)
        if not a or a["feedback"] == value or value not in (1, -1):
            return
        await asyncio.to_thread(self.store.set_feedback, alert_id, value)
        vec = await asyncio.to_thread(self.embedder.passages, [a["text"] or a["reason"]])
        for key in json.loads(a["rules"]):
            r = self.rules.get(key)
            if not r:
                continue
            if a["text"]:
                await asyncio.to_thread(self.store.add_example, key, value, a["text"], vec[0])
            step = STEP.get(key, 0.005)
            lo, hi = BOUNDS.get(key, SIM_BOUNDS)
            new = r["threshold"] + (step if value < 0 else -step / 2)
            await asyncio.to_thread(self.store.update_rule, key, threshold=min(hi, max(lo, new)))
        self.reload()
        await self.broadcast(self.view(await asyncio.to_thread(self.store.alert, alert_id)))
        await self.broadcast({"type": "alert_rules", "rules": self.rules_view()})

"""Who is speaking: teacher (T) / student (S) / unknown (?).

Three kinds of cues are combined into a probability P(teacher):
  1. Voice: speaker embeddings (sherpa-onnx) clustered live. The teacher is the cluster
     that matches the subject's saved voiceprint or, if there is none, the one who talks most.
  2. Text: phrases typical of a teacher / a student (Spanish and English patterns).
  3. Turns: long sentences -> teacher; a teacher's "good question" -> the previous one was a student.
Every new sentence re-scores the whole class (cheap) and the label changes are returned.
"""
import math
import re

import numpy as np

SR = 16000

# Regex cues. They match the *spoken* language, so they stay in Spanish (plus English).
TEACHER_CUES = [
    r"\babrid\b", r"\bapuntad\b", r"\bfijaos\b", r"\bmirad\b", r"\bhaced\b", r"\bsacad\b",
    r"\bentra en el examen\b", r"\bpara el (examen|parcial|jueves|lunes|martes|miércoles|viernes)\b",
    r"\bvamos a (ver|empezar|hacer)\b", r"\bhoy (vamos|veremos)\b", r"\bpor ejemplo\b",
    r"\bbuena pregunta\b", r"\bmuy bien\b", r"\bexacto\b", r"\bformalmente\b",
    r"\blet'?s (see|start)\b", r"\bopen your\b", r"\bfor the exam\b", r"\bgood question\b",
]
STUDENT_CUES = [
    r"\bprofe\b", r"\bprofesora?\b", r"\bno (lo )?entiendo\b", r"\buna (pregunta|duda)\b",
    r"\b¿?esto entra\b", r"\bpuedes repetir\b", r"\bpodría repetir\b", r"\bme he perdido\b",
    r"\bteacher\b", r"\bi don'?t understand\b", r"\bquick question\b",
]
ANSWERS_STUDENT = re.compile(
    r"^\s*(buena pregunta|muy bien|exacto|efectivamente|correcto|dime|claro|good question|exactly)\b",
    re.I)


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def text_score(text: str) -> float:
    """Log-odds contribution of the text: > 0 sounds like the teacher, < 0 like a student."""
    t = text.lower()
    teacher = sum(1.0 for p in TEACHER_CUES if re.search(p, t))
    student = sum(1.0 for p in STUDENT_CUES if re.search(p, t))
    s = min(teacher, 2) * 1.0 - min(student, 2) * 1.5
    n = len(t.split())
    if n > 25:
        s += 0.7
    elif n < 12 and t.rstrip().endswith("?"):
        s -= 0.4  # weak: teachers also ask rhetorical questions
    return s


class SpeakerTracker:
    def __init__(self, model_onnx: str, cfg: dict, teacher_voiceprint: np.ndarray | None = None):
        import sherpa_onnx as so
        self.extractor = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(model=model_onnx, num_threads=2))
        self.same_threshold = cfg["same_speaker_threshold"]
        self.voiceprint_threshold = cfg["saved_teacher_threshold"]
        self.min_speech = cfg["min_speech_to_decide_s"]
        self.voiceprint = teacher_voiceprint
        self.clusters: list[dict] = []   # {c: centroid, n, dur}
        self.sentences: list[dict] = []  # {id, cluster, emb, dur, text_score, label, p, answers}

    # --- voice
    def _embedding(self, audio):
        s = self.extractor.create_stream()
        s.accept_waveform(SR, audio)
        s.input_finished()
        e = np.array(self.extractor.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def _assign_cluster(self, e, dur):
        if self.clusters:
            sims = [float(c["c"] @ e) for c in self.clusters]
            i = int(np.argmax(sims))
            if sims[i] >= self.same_threshold:
                c = self.clusters[i]
                total = c["c"] * c["n"] + e
                c["c"] = total / np.linalg.norm(total)
                c["n"] += 1
                c["dur"] += dur
                return i
        self.clusters.append({"c": e, "n": 1, "dur": dur})
        return len(self.clusters) - 1

    def teacher_cluster(self) -> tuple[int | None, float]:
        """(index of the teacher's cluster, confidence 0..1)."""
        if not self.clusters:
            return None, 0.0
        if self.voiceprint is not None:
            sims = [float(c["c"] @ self.voiceprint) for c in self.clusters]
            i = int(np.argmax(sims))
            if sims[i] >= self.voiceprint_threshold:
                return i, 1.0
        total = sum(c["dur"] for c in self.clusters)
        i = int(np.argmax([c["dur"] for c in self.clusters]))
        share = self.clusters[i]["dur"] / max(total, 1e-6)
        if total >= self.min_speech and share >= 0.4:
            return i, 1.0
        return i, 0.5 * min(1.0, total / self.min_speech)  # provisional

    # --- combination
    def _rescore(self):
        tc, conf = self.teacher_cluster()
        changes = []
        for k, s in enumerate(self.sentences):
            voice = 0.0
            if tc is not None and conf > 0:
                if s["emb"] is not None:
                    voice = 12 * (float(self.clusters[tc]["c"] @ s["emb"]) - 0.65) * conf
                elif k > 0:  # very short sentence: assume continuity with the previous one
                    voice = 0.5 * (1 if self.sentences[k - 1]["label"] == "T" else -1)
            text = s["text_score"]
            if k + 1 < len(self.sentences) and self.sentences[k + 1]["answers"]:
                text -= 1.5  # the teacher answered this sentence -> probably a student
            p = _sigmoid(voice + text)
            label = "T" if p >= 0.65 else "S" if p <= 0.35 else "?"
            if label != s["label"] or abs(p - s["p"]) > 0.05:
                if label != s["label"] and s["label"] is not None:
                    changes.append((s["id"], label, p))
                s["label"], s["p"] = label, p
        return changes

    def add_sentence(self, temp_id, audio: np.ndarray, text: str):
        """Returns (cluster, label, p_teacher, label changes in earlier sentences)."""
        dur = len(audio) / SR
        emb, cluster = None, None
        if dur >= 0.6:
            emb = self._embedding(audio)
            cluster = self._assign_cluster(emb, dur)
        s = {"id": temp_id, "cluster": cluster, "emb": emb, "dur": dur,
             "text_score": text_score(text), "label": None, "p": 0.5,
             "answers": bool(ANSWERS_STUDENT.search(text))}
        self.sentences.append(s)
        changes = self._rescore()
        return cluster, s["label"], s["p"], [c for c in changes if c[0] != temp_id]

    def set_id(self, temp_id, real_id):
        for s in self.sentences:
            if s["id"] == temp_id:
                s["id"] = real_id

    def teacher_centroid(self):
        tc, conf = self.teacher_cluster()
        if tc is None or conf < 1.0:
            return None, 0
        return self.clusters[tc]["c"], self.clusters[tc]["n"]

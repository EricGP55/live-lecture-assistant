"""Quién habla: profesor (P) / alumno (A) / desconocido (?).

Combina tres pistas en una probabilidad P(profesor):
  1. Voz: huella (embedding de hablante, sherpa-onnx) agrupada en vivo; el profesor es
     el grupo que coincide con la huella guardada de la asignatura o, si no hay,
     el que más habla.
  2. Texto: frases típicas de profesor / alumno.
  3. Turnos: frases largas -> profesor; "buena pregunta" del profe -> la anterior era un alumno.
Cada frase nueva recalcula las etiquetas de toda la clase (barato) y se devuelven los cambios.
"""
import math
import re

import numpy as np

SR = 16000

PISTAS_PROFE = [
    r"\babrid\b", r"\bapuntad\b", r"\bfijaos\b", r"\bmirad\b", r"\bhaced\b", r"\bsacad\b",
    r"\bentra en el examen\b", r"\bpara el (examen|parcial|jueves|lunes|martes|miércoles|viernes)\b",
    r"\bvamos a (ver|empezar|hacer)\b", r"\bhoy (vamos|veremos)\b", r"\bpor ejemplo\b",
    r"\bbuena pregunta\b", r"\bmuy bien\b", r"\bexacto\b", r"\bformalmente\b",
    r"\blet'?s (see|start)\b", r"\bopen your\b", r"\bfor the exam\b", r"\bgood question\b",
]
PISTAS_ALUMNO = [
    r"\bprofe\b", r"\bprofesora?\b", r"\bno (lo )?entiendo\b", r"\buna (pregunta|duda)\b",
    r"\b¿?esto entra\b", r"\bpuedes repetir\b", r"\bpodría repetir\b", r"\bme he perdido\b",
    r"\bteacher\b", r"\bi don'?t understand\b", r"\bquick question\b",
]
RESPUESTA_A_ALUMNO = re.compile(
    r"^\s*(buena pregunta|muy bien|exacto|efectivamente|correcto|dime|claro|good question|exactly)\b",
    re.I)


def _sig(x):
    return 1 / (1 + math.exp(-x))


def puntuacion_texto(texto: str) -> float:
    t = texto.lower()
    lp = sum(1.0 for p in PISTAS_PROFE if re.search(p, t))
    la = sum(1.0 for p in PISTAS_ALUMNO if re.search(p, t))
    s = min(lp, 2) * 1.0 - min(la, 2) * 1.5
    n = len(t.split())
    if n > 25:
        s += 0.7
    elif n < 12 and t.rstrip().endswith("?"):
        s -= 0.4  # débil: el profe también hace preguntas retóricas
    return s


class Voces:
    def __init__(self, modelo_onnx: str, cfg: dict, huella_profe: np.ndarray | None = None):
        import sherpa_onnx as so
        self.ex = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(model=modelo_onnx, num_threads=2))
        self.umbral = cfg["umbral_mismo_hablante"]
        self.umbral_huella = cfg["umbral_profe_guardado"]
        self.voz_min = cfg["voz_min_para_decidir_s"]
        self.huella = huella_profe
        self.grupos: list[dict] = []   # {c: centroide, n, dur}
        self.frases: list[dict] = []   # {id, g, emb, dur, lt, label, p}

    # --- voz
    def _embedding(self, audio):
        s = self.ex.create_stream()
        s.accept_waveform(SR, audio)
        s.input_finished()
        e = np.array(self.ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def _agrupar(self, e, dur):
        if self.grupos:
            sims = [float(g["c"] @ e) for g in self.grupos]
            i = int(np.argmax(sims))
            if sims[i] >= self.umbral:
                g = self.grupos[i]
                c = g["c"] * g["n"] + e
                g["c"] = c / np.linalg.norm(c)
                g["n"] += 1
                g["dur"] += dur
                return i
        self.grupos.append({"c": e, "n": 1, "dur": dur})
        return len(self.grupos) - 1

    def grupo_profe(self) -> tuple[int | None, float]:
        """(índice del grupo del profesor, confianza 0..1)."""
        if not self.grupos:
            return None, 0.0
        if self.huella is not None:
            sims = [float(g["c"] @ self.huella) for g in self.grupos]
            i = int(np.argmax(sims))
            if sims[i] >= self.umbral_huella:
                return i, 1.0
        total = sum(g["dur"] for g in self.grupos)
        i = int(np.argmax([g["dur"] for g in self.grupos]))
        cuota = self.grupos[i]["dur"] / max(total, 1e-6)
        if total >= self.voz_min and cuota >= 0.4:
            return i, 1.0
        return i, 0.5 * min(1.0, total / self.voz_min)  # provisional

    # --- combinación
    def _recalcular(self):
        gp, conf = self.grupo_profe()
        cambios = []
        for k, f in enumerate(self.frases):
            lv = 0.0
            if gp is not None and conf > 0:
                if f["emb"] is not None:
                    lv = 12 * (float(self.grupos[gp]["c"] @ f["emb"]) - 0.65) * conf
                elif k > 0:  # frase muy corta: continuidad con la anterior
                    lv = 0.5 * (1 if self.frases[k - 1]["label"] == "P" else -1)
            lt = f["lt"]
            if k + 1 < len(self.frases) and self.frases[k + 1].get("responde"):
                lt -= 1.5  # el profe ha respondido a esta frase -> probablemente alumno
            p = _sig(lv + lt)
            label = "P" if p >= 0.65 else "A" if p <= 0.35 else "?"
            if label != f["label"] or abs(p - f["p"]) > 0.05:
                if label != f["label"] and f["label"] is not None:
                    cambios.append((f["id"], label, p))
                f["label"], f["p"] = label, p
        return cambios

    def nueva_frase(self, frase_id_provisional, audio: np.ndarray, texto: str):
        """Devuelve (grupo, label, p_profe, cambios_en_frases_anteriores)."""
        dur = len(audio) / SR
        emb, g = None, None
        if dur >= 0.6:
            emb = self._embedding(audio)
            g = self._agrupar(emb, dur)
        f = {"id": frase_id_provisional, "g": g, "emb": emb, "dur": dur,
             "lt": puntuacion_texto(texto), "label": None, "p": 0.5,
             "responde": bool(RESPUESTA_A_ALUMNO.search(texto))}
        self.frases.append(f)
        cambios = self._recalcular()
        return g, f["label"], f["p"], [c for c in cambios if c[0] != frase_id_provisional]

    def set_id(self, provisional, real):
        for f in self.frases:
            if f["id"] == provisional:
                f["id"] = real

    def centroide_profe(self):
        gp, conf = self.grupo_profe()
        if gp is None or conf < 1.0:
            return None, 0
        return self.grupos[gp]["c"], self.grupos[gp]["n"]

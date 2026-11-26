"""Bucle principal: audio -> VAD -> transcripción + voz -> base de datos -> eventos para la web.

Todo el trabajo con MLX y ONNX ocurre en este único hilo (MLX no es thread-safe).
"""
import threading
import time
from datetime import datetime

import numpy as np

from .audio import SR, Archivo, Grabadora, Micro
from .config import RAIZ
from .store import Store
from .stt import Transcriptor
from .voces import Voces


class Pipeline(threading.Thread):
    def __init__(self, cfg: dict, store: Store, asignatura: str, emitir, archivo=None, rapido=False):
        super().__init__(daemon=True)
        self.cfg, self.store, self.emitir = cfg, store, emitir
        self.asig = store.asignatura(asignatura)
        self.archivo, self.rapido = archivo, rapido
        self._parar = threading.Event()
        self.clase_id = None
        self.estado = "Arrancando…"
        self.listo = threading.Event()

    def parar(self):
        self._parar.set()

    def _estado(self, texto):
        self.estado = texto
        self.emitir({"tipo": "estado", "texto": texto})

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as e:  # que el error se vea en la web, no solo en la terminal
            self._estado(f"Error: {e}")
            raise
        finally:
            self.listo.set()

    def _run(self):
        cfg = self.cfg
        self._estado("Cargando modelo de transcripción…")
        stt = Transcriptor(cfg["stt"]["modelo"])
        self._estado("Cargando modelos de voz…")
        import sherpa_onnx as so
        vc = so.VadModelConfig()
        vc.silero_vad.model = str(RAIZ / "models/silero_vad.onnx")
        vc.silero_vad.min_silence_duration = cfg["vad"]["silencio_min_s"]
        vc.silero_vad.min_speech_duration = cfg["vad"]["voz_min_s"]
        vc.silero_vad.max_speech_duration = cfg["vad"]["frase_max_s"]
        vc.sample_rate = SR
        vad = so.VoiceActivityDetector(vc, buffer_size_in_seconds=120)
        voces = Voces(str(RAIZ / "models/speaker.onnx"), cfg["voces"],
                      self.store.huella_profe(self.asig["id"]))

        fuente = Archivo(self.archivo, tiempo_real=not self.rapido) if self.archivo \
            else Micro(cfg["audio"]["dispositivo"])

        self.clase_id = self.store.nueva_clase(self.asig["id"], cfg.get("idioma", "es"))
        self.inicio = time.time()
        grab = None
        if cfg["audio"]["guardar"]:
            nombre = f"{self.asig['nombre']}_{datetime.now():%Y-%m-%d_%H%M}.opus".replace(" ", "_")
            ruta = cfg["datos"] / "audio" / nombre
            grab = Grabadora(ruta, cfg["audio"]["bitrate"])
            self.store.set_audio(self.clase_id, str(ruta))
        self.emitir({"tipo": "clase", "clase_id": self.clase_id, "asignatura": self.asig["nombre"],
                     "inicio": self.inicio, "fuente": fuente.nombre,
                     "huella": voces.huella is not None})
        self._estado(f"Escuchando · {fuente.nombre}")

        # anillo con los últimos 30 s, para los parciales
        anillo = np.zeros(SR * 30, dtype=np.float32)
        n = 0                    # muestras totales consumidas
        voz_desde = None         # muestra donde empezó la voz en curso
        ult_parcial = 0.0
        ult_nivel = 0.0
        cada = cfg["stt"]["parcial_cada_s"]

        ult_fin = [0]  # fin (en muestras) de la última frase

        def antes_de(muestra, largo):
            """Audio del anillo justo antes de `muestra` (pre-roll, recupera inicios de palabra)."""
            ini = max(0, muestra - largo, n - len(anillo))
            if ini >= muestra:
                return np.zeros(0, dtype=np.float32)
            return anillo[np.arange(ini, muestra) % len(anillo)]

        def procesar_segmento(seg_start, samples):
            t_proc = time.monotonic()
            # el VAD detecta la voz ~0,5 s tarde: recuperamos hasta 0,7 s sin invadir la frase anterior
            pre = antes_de(seg_start, min(int(0.7 * SR), max(0, seg_start - ult_fin[0])))
            seg_start -= len(pre)
            samples = np.concatenate([pre, samples])
            ult_fin[0] = seg_start + len(samples)
            texto = stt.transcribir(samples)
            if not texto:
                return
            g, label, p, cambios = voces.nueva_frase(-1, samples, texto)
            t0, t1 = seg_start / SR, (seg_start + len(samples)) / SR
            fid = self.store.nueva_frase(self.clase_id, t0, t1, texto, label, p, g)
            voces.set_id(-1, fid)
            lat = int((time.monotonic() - t_proc) * 1000)
            self.emitir({"tipo": "frase", "id": fid, "t0": t0, "t1": t1, "texto": texto,
                         "hablante": label, "p": round(p, 2), "proc_ms": lat})
            for cid, clabel, cp in cambios:
                self.store.set_hablante(cid, clabel, cp)
                self.emitir({"tipo": "hablante", "id": cid, "hablante": clabel, "p": round(cp, 2)})

        while True:
            if self._parar.is_set() or (fuente.terminado and fuente.q.empty()):
                break
            x = fuente.leer(timeout=0.3)
            if x is None:
                continue
            if grab:
                grab.escribir(x)
            # anillo
            k = len(x)
            pos = n % len(anillo)
            fin = min(k, len(anillo) - pos)
            anillo[pos:pos + fin] = x[:fin]
            anillo[:k - fin] = x[fin:]
            n += k

            vad.accept_waveform(x)
            hablando = vad.is_speech_detected()
            if hablando and voz_desde is None:
                voz_desde = max(0, n - int(0.4 * SR))
            while not vad.empty():
                s = vad.front
                procesar_segmento(s.start, np.array(s.samples, dtype=np.float32))
                vad.pop()
                voz_desde = None if not vad.is_speech_detected() else max(0, n - int(0.4 * SR))
                self.emitir({"tipo": "parcial", "texto": ""})

            ahora = time.monotonic()
            if ahora - ult_nivel > 0.25:
                ult_nivel = ahora
                self.emitir({"tipo": "nivel", "rms": float(np.sqrt(np.mean(x ** 2))),
                             "t": n / SR, "retraso_s": round(fuente.pendiente_s(), 2)})
            # parcial: re-transcribe la frase en curso si no vamos con retraso
            if (hablando and voz_desde is not None and ahora - ult_parcial >= cada
                    and fuente.pendiente_s() < 0.5 and n - voz_desde > SR):
                ult_parcial = ahora
                largo = min(n - voz_desde, len(anillo))
                idx = (np.arange(n - largo, n)) % len(anillo)
                texto = stt.transcribir(anillo[idx])
                if texto:
                    self.emitir({"tipo": "parcial", "texto": texto, "t": voz_desde / SR})

        # cierre
        vad.flush()
        while not vad.empty():
            s = vad.front
            procesar_segmento(s.start, np.array(s.samples, dtype=np.float32))
            vad.pop()
        fuente.parar()
        if grab:
            grab.cerrar()
        c, cn = voces.centroide_profe()
        if c is not None:
            self.store.guardar_huella_profe(self.asig["id"], c, cn)
        self.store.cerrar_clase(self.clase_id)
        self.emitir({"tipo": "parcial", "texto": ""})
        self._estado("Clase terminada" + (" · huella del profesor guardada" if c is not None else ""))

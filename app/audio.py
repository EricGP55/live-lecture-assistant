"""Fuentes de audio (micro o archivo) a 16 kHz mono float32, y grabación a Opus."""
import queue
import subprocess
import threading
import time

import numpy as np

SR = 16000
BLOQUE = 512  # 32 ms, lo que espera el VAD silero


def listar_dispositivos() -> list[dict]:
    import sounddevice as sd
    return [{"indice": i, "nombre": d["name"], "canales": d["max_input_channels"]}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def _resolver(dispositivo):
    if dispositivo is None or isinstance(dispositivo, int):
        return dispositivo
    if str(dispositivo).isdigit():
        return int(dispositivo)
    for d in listar_dispositivos():
        if str(dispositivo).lower() in d["nombre"].lower():
            return d["indice"]
    raise SystemExit(f"No encuentro el micro '{dispositivo}'. Usa --dispositivos para ver la lista.")


class Fuente:
    """Interfaz: leer(timeout) -> np.ndarray | None ; parar()."""
    q: queue.Queue

    def leer(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def pendiente_s(self) -> float:
        return self.q.qsize() * BLOQUE / SR


class Micro(Fuente):
    def __init__(self, dispositivo=None):
        import sounddevice as sd
        self.q = queue.Queue()
        self.terminado = False
        dev = _resolver(dispositivo)
        self.nombre = sd.query_devices(dev, "input")["name"]
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                     blocksize=BLOQUE, device=dev, callback=self._cb)
        self.stream.start()

    def _cb(self, data, frames, t, status):
        self.q.put(data[:, 0].copy())

    def parar(self):
        self.stream.stop()
        self.stream.close()
        self.terminado = True


class Archivo(Fuente):
    """Lee un archivo de audio (para pruebas). tiempo_real=True simula una clase en directo."""

    def __init__(self, ruta, tiempo_real=True):
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(ruta), "-f", "f32le",
                              "-ac", "1", "-ar", str(SR), "pipe:1"], capture_output=True, check=True)
        self.audio = np.frombuffer(out.stdout, dtype=np.float32)
        self.nombre = f"archivo:{ruta}"
        self.q = queue.Queue(maxsize=0 if tiempo_real else 64)
        self.terminado = False
        self._stop = False
        threading.Thread(target=self._run, args=(tiempo_real,), daemon=True).start()

    def _run(self, tiempo_real):
        t0 = time.monotonic()
        for i in range(0, len(self.audio), BLOQUE):
            if self._stop:
                break
            if tiempo_real:
                espera = t0 + i / SR - time.monotonic()
                if espera > 0:
                    time.sleep(espera)
            self.q.put(self.audio[i:i + BLOQUE])
        # 1 s de silencio para cerrar la última frase
        for _ in range(SR // BLOQUE):
            self.q.put(np.zeros(BLOQUE, dtype=np.float32))
        self.terminado = True

    def parar(self):
        self._stop = True


class Grabadora:
    """Comprime a Opus en segundo plano con ffmpeg (ocupa ~11 MB/h a 24 kbps)."""

    def __init__(self, ruta, bitrate="24k"):
        self.ruta = str(ruta)
        self.p = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
             "-i", "pipe:0", "-c:a", "libopus", "-b:a", bitrate, "-application", "voip", self.ruta],
            stdin=subprocess.PIPE)

    def escribir(self, x: np.ndarray):
        try:
            self.p.stdin.write(x.astype(np.float32).tobytes())
        except BrokenPipeError:
            pass

    def cerrar(self):
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.wait(timeout=10)

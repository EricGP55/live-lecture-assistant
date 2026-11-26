"""Audio sources (mic or file) at 16 kHz mono float32, plus Opus recording."""
import queue
import subprocess
import threading
import time

import numpy as np

SR = 16000
BLOCK = 512  # 32 ms, what the silero VAD expects


def list_devices() -> list[dict]:
    import sounddevice as sd
    return [{"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def _resolve(device):
    if device is None or isinstance(device, int):
        return device
    if str(device).isdigit():
        return int(device)
    for d in list_devices():
        if str(device).lower() in d["name"].lower():
            return d["index"]
    raise SystemExit(f"Microphone '{device}' not found. Use --devices to see the list.")


class Source:
    """Interface: read(timeout) -> np.ndarray | None ; stop()."""
    q: queue.Queue

    def read(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def backlog_s(self) -> float:
        return self.q.qsize() * BLOCK / SR


class Microphone(Source):
    def __init__(self, device=None):
        import sounddevice as sd
        self.q = queue.Queue()
        self.finished = False
        dev = _resolve(device)
        self.name = sd.query_devices(dev, "input")["name"]
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                     blocksize=BLOCK, device=dev, callback=self._callback)
        self.stream.start()

    def _callback(self, data, frames, t, status):
        self.q.put(data[:, 0].copy())

    def stop(self):
        self.stream.stop()
        self.stream.close()
        self.finished = True


class AudioFile(Source):
    """Reads an audio file (for testing). realtime=True simulates a live class."""

    def __init__(self, path, realtime=True):
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le",
                              "-ac", "1", "-ar", str(SR), "pipe:1"], capture_output=True, check=True)
        self.audio = np.frombuffer(out.stdout, dtype=np.float32)
        self.name = f"file:{path}"
        self.q = queue.Queue(maxsize=0 if realtime else 64)
        self.finished = False
        self._stop = False
        threading.Thread(target=self._run, args=(realtime,), daemon=True).start()

    def _run(self, realtime):
        t0 = time.monotonic()
        for i in range(0, len(self.audio), BLOCK):
            if self._stop:
                break
            if realtime:
                wait = t0 + i / SR - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self.q.put(self.audio[i:i + BLOCK])
        # 1 s of silence to close the last sentence
        for _ in range(SR // BLOCK):
            self.q.put(np.zeros(BLOCK, dtype=np.float32))
        self.finished = True

    def stop(self):
        self._stop = True


class Recorder:
    """Compresses to Opus in the background with ffmpeg (~11 MB/h at 24 kbps)."""

    def __init__(self, path, bitrate="24k"):
        self.path = str(path)
        self.p = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
             "-i", "pipe:0", "-c:a", "libopus", "-b:a", bitrate, "-application", "voip", self.path],
            stdin=subprocess.PIPE)

    def write(self, x: np.ndarray):
        try:
            self.p.stdin.write(x.astype(np.float32).tobytes())
        except BrokenPipeError:
            pass

    def close(self):
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.wait(timeout=10)

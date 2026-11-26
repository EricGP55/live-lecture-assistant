"""Main loop: audio -> VAD -> transcription + voice -> database -> events for the web page.

All MLX and ONNX work happens on this single thread (MLX is not thread-safe).
"""
import threading
import time
from datetime import datetime

import numpy as np

from .audio import SR, AudioFile, Microphone, Recorder
from .config import ROOT
from .speakers import SpeakerTracker
from .store import Store
from .stt import Transcriber


class Pipeline(threading.Thread):
    def __init__(self, cfg: dict, store: Store, subject: str, emit, audio_file=None, fast=False,
                 embedder=None):
        super().__init__(daemon=True)
        self.cfg, self.store, self.emit = cfg, store, emit
        self.subject = store.subject(subject)
        self.audio_file, self.fast = audio_file, fast
        self.embedder = embedder  # optional: embeds each sentence for search (phase 2)
        self.t_now = 0.0
        self._stop_event = threading.Event()
        self.class_id = None
        self.status = "Starting…"

    def current_time(self) -> float:
        """Class time in seconds (audio consumed so far)."""
        return self.t_now

    def stop(self):
        self._stop_event.set()

    def _status(self, text):
        self.status = text
        self.emit({"type": "status", "text": text})

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as e:  # show the error on the page, not only in the terminal
            self._status(f"Error: {e}")
            raise

    def _run(self):
        cfg = self.cfg
        self._status("Loading speech-to-text model…")
        stt = Transcriber(cfg["stt"]["model"])
        self._status("Loading voice models…")
        import sherpa_onnx as so
        vc = so.VadModelConfig()
        vc.silero_vad.model = str(ROOT / "models/silero_vad.onnx")
        vc.silero_vad.min_silence_duration = cfg["vad"]["min_silence_s"]
        vc.silero_vad.min_speech_duration = cfg["vad"]["min_speech_s"]
        vc.silero_vad.max_speech_duration = cfg["vad"]["max_sentence_s"]
        vc.sample_rate = SR
        vad = so.VoiceActivityDetector(vc, buffer_size_in_seconds=120)
        speakers = SpeakerTracker(str(ROOT / "models/speaker.onnx"), cfg["speakers"],
                                  self.store.teacher_voiceprint(self.subject["id"]))

        source = AudioFile(self.audio_file, realtime=not self.fast) if self.audio_file \
            else Microphone(cfg["audio"]["device"])

        self.class_id = self.store.new_class(self.subject["id"], cfg.get("language", "es"))
        self.started_at = time.time()
        recorder = None
        if cfg["audio"]["save"]:
            name = f"{self.subject['name']}_{datetime.now():%Y-%m-%d_%H%M}.opus".replace(" ", "_")
            path = cfg["data"] / "audio" / name
            recorder = Recorder(path, cfg["audio"]["bitrate"])
            self.store.set_audio(self.class_id, str(path))
        self.emit({"type": "class", "class_id": self.class_id, "subject": self.subject["name"],
                   "started_at": self.started_at, "source": source.name,
                   "voiceprint": speakers.voiceprint is not None})
        self._status(f"Listening · {source.name}")

        ring = np.zeros(SR * 30, dtype=np.float32)  # last 30 s, for partials and pre-roll
        n = 0                     # total samples consumed
        speech_from = None        # sample where the current speech started
        last_partial = 0.0
        last_level = 0.0
        partial_every = cfg["stt"]["partial_every_s"]
        last_end = [0]            # end (in samples) of the last sentence
        pause_after = int(cfg.get("doubts", {}).get("pause_moment_s", 4) * SR)
        pause_sent = [True]       # one "pause" event per silence

        def audio_before(sample, length):
            """Ring audio right before `sample` (pre-roll, recovers word onsets)."""
            start = max(0, sample - length, n - len(ring))
            if start >= sample:
                return np.zeros(0, dtype=np.float32)
            return ring[np.arange(start, sample) % len(ring)]

        def process_segment(seg_start, samples):
            t_proc = time.monotonic()
            # the VAD fires ~0.5 s late: recover up to 0.7 s without reaching into the previous sentence
            pre = audio_before(seg_start, min(int(0.7 * SR), max(0, seg_start - last_end[0])))
            seg_start -= len(pre)
            samples = np.concatenate([pre, samples])
            last_end[0] = seg_start + len(samples)
            pause_sent[0] = False
            text = stt.transcribe(samples)
            if not text:
                return
            cluster, label, p, changes = speakers.add_sentence(-1, samples, text)
            t0, t1 = seg_start / SR, (seg_start + len(samples)) / SR
            sid = self.store.new_sentence(self.class_id, t0, t1, text, label, p, cluster)
            speakers.set_id(-1, sid)
            if self.embedder is not None:
                self.store.set_embedding(sid, self.embedder.passages([text])[0])
            proc_ms = int((time.monotonic() - t_proc) * 1000)
            self.emit({"type": "sentence", "id": sid, "t0": t0, "t1": t1, "text": text,
                       "speaker": label, "p": round(p, 2), "proc_ms": proc_ms})
            for cid, clabel, cp in changes:
                self.store.set_speaker(cid, clabel, cp)
                self.emit({"type": "speaker", "id": cid, "speaker": clabel, "p": round(cp, 2)})

        while True:
            if self._stop_event.is_set() or (source.finished and source.q.empty()):
                break
            x = source.read(timeout=0.3)
            if x is None:
                continue
            if recorder:
                recorder.write(x)
            k = len(x)
            pos = n % len(ring)
            first = min(k, len(ring) - pos)
            ring[pos:pos + first] = x[:first]
            ring[:k - first] = x[first:]
            n += k
            self.t_now = n / SR

            vad.accept_waveform(x)
            speaking = vad.is_speech_detected()
            if speaking and speech_from is None:
                speech_from = max(0, n - int(0.4 * SR))
            while not vad.empty():
                s = vad.front
                process_segment(s.start, np.array(s.samples, dtype=np.float32))
                vad.pop()
                speech_from = max(0, n - int(0.4 * SR)) if vad.is_speech_detected() else None
                self.emit({"type": "partial", "text": ""})

            # long silence after someone spoke: a possible moment to ask (used by the doubts mode)
            if not speaking and not pause_sent[0] and n - last_end[0] >= pause_after:
                pause_sent[0] = True
                self.emit({"type": "pause", "after_s": (n - last_end[0]) / SR, "t": n / SR})

            now = time.monotonic()
            if now - last_level > 0.25:
                last_level = now
                self.emit({"type": "level", "rms": float(np.sqrt(np.mean(x ** 2))),
                           "t": n / SR, "lag_s": round(source.backlog_s(), 2)})
            # partial: re-transcribe the sentence in progress, unless we are lagging behind
            if (speaking and speech_from is not None and now - last_partial >= partial_every
                    and source.backlog_s() < 0.5 and n - speech_from > SR):
                last_partial = now
                length = min(n - speech_from, len(ring))
                text = stt.transcribe(ring[np.arange(n - length, n) % len(ring)])
                if text:
                    self.emit({"type": "partial", "text": text, "t": speech_from / SR})

        # wrap up
        vad.flush()
        while not vad.empty():
            s = vad.front
            process_segment(s.start, np.array(s.samples, dtype=np.float32))
            vad.pop()
        source.stop()
        if recorder:
            recorder.close()
        centroid, cn = speakers.teacher_centroid()
        if centroid is not None:
            self.store.save_teacher_voiceprint(self.subject["id"], centroid, cn)
        self.store.close_class(self.class_id)
        self.emit({"type": "partial", "text": ""})
        self._status("Class finished" + (" · teacher voiceprint saved" if centroid is not None else ""))
